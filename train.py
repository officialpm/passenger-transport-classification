# Copyright 2026 Parth Maniar. Apache-2.0.
import os, time, json, hashlib
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix
from catboost import CatBoostClassifier
import lightgbm as lgb
import argparse
parser=argparse.ArgumentParser(description='Passenger transport classification with fold-safe validation')
parser.add_argument('--data-dir',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,default=Path('outputs'))
args=parser.parse_args()
root=args.data_dir;W=args.output_dir;W.mkdir(parents=True,exist_ok=True)
train=pd.read_csv(root/'train.csv'); test=pd.read_csv(root/'test.csv'); sample=pd.read_csv(root/'sample_submission.csv')
assert 'Transported' in train and train.PassengerId.is_unique and test.PassengerId.is_unique
assert sample.PassengerId.tolist()==test.PassengerId.tolist()
y=train.Transported.astype(int).to_numpy(); groups=train.PassengerId.str.split('_').str[0]
print('Shapes',train.shape,test.shape,'target rate',y.mean())
print(train.isna().mean().sort_values(ascending=False))
train.drop(columns='Transported').isna().mean().sort_values().plot.barh(figsize=(7,4),title='Missing values before preprocessing');plt.tight_layout();plt.savefig(W/'missingness.png');plt.show()


# Copyright 2026 Parth Maniar. Apache-2.0.
SPEND=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
CATS=['HomePlanet','Destination','Deck','Side','CryoSleep','VIP']
all_cov=pd.concat([train.drop(columns='Transported'),test],ignore_index=True)
group_counts=all_cov.PassengerId.str.split('_').str[0].value_counts()
surname_counts=all_cov.Name.str.split().str[-1].value_counts()
def features(frame):
    d=frame.drop(columns=['Transported'],errors='ignore').copy()
    cab=d.Cabin.str.split('/',expand=True).reindex(columns=[0,1,2])
    d['Deck']=cab[0];d['Side']=cab[2];d['CabNum']=pd.to_numeric(cab[1],errors='coerce')
    d['CabRegion']=d.CabNum//300
    d['GroupSize']=d.PassengerId.str.split('_').str[0].map(group_counts)
    d['FamilySize']=d.Name.str.split().str[-1].map(surname_counts).fillna(1)
    d['Solo']=(d.GroupSize==1).astype(int)
    for c in SPEND:
        d[c+'_Missing']=d[c].isna().astype(int)
        d.loc[d.CryoSleep.eq(True)&d[c].isna(),c]=0
        d['Log_'+c]=np.log1p(d[c])
    d['KnownSpend']=d[SPEND].sum(axis=1,min_count=1)
    d['SpendObserved']=d[SPEND].notna().sum(axis=1)
    d['ZeroObservedSpend']=(d.KnownSpend.eq(0)&d.SpendObserved.eq(5)).astype(int)
    d['AgeMissing']=d.Age.isna().astype(int)
    d['AgeBin']=pd.cut(d.Age,[-1,12,17,25,40,60,200],labels=False)
    for c in CATS:d[c]=d[c].fillna('Missing').astype(str)
    return d.drop(columns=['PassengerId','Name','Cabin'])
X=features(train);XT=features(test)
NUM=[c for c in X if c not in CATS]
def fold_frames(a,b,c):
    median=a[NUM].median().fillna(0)
    return [z.assign(**{k:z[k].fillna(median[k]) for k in NUM}) for z in (a,b,c)]
def lgb_frame(z):
    z=z.copy()
    # Fixed categories come from unlabeled covariates, not targets or frequencies.
    for c in CATS:z[c]=pd.Categorical(z[c],categories=sorted(set(X[c])|set(XT[c])))
    return z
print('Features',list(X.columns))


# Copyright 2026 Parth Maniar. Apache-2.0.
start=time.monotonic();folds=list(StratifiedKFold(5,shuffle=True,random_state=2026).split(X,y))
oof={k:np.zeros(len(X)) for k in ['cat','lgb']};pred={k:np.zeros(len(XT)) for k in oof}; rows=[];importances=[]
for f,(tr,va) in enumerate(folds):
    assert time.monotonic()-start<90*60,'Time guard; do not save partial predictions'
    a,b,c=fold_frames(X.iloc[tr].copy(),X.iloc[va].copy(),XT.copy())
    cat=CatBoostClassifier(iterations=1800,depth=6,learning_rate=.035,l2_leaf_reg=6,loss_function='Logloss',eval_metric='Logloss',random_seed=2026+f,thread_count=4,verbose=False,allow_writing_files=False)
    cat.fit(a,y[tr],cat_features=CATS,eval_set=(b,y[va]),early_stopping_rounds=150)
    oof['cat'][va]=cat.predict_proba(b)[:,1];pred['cat']+=cat.predict_proba(c)[:,1]/5
    gb=lgb.LGBMClassifier(n_estimators=2200,learning_rate=.025,num_leaves=24,min_child_samples=35,colsample_bytree=.85,reg_lambda=4,n_jobs=4,verbosity=-1,random_state=2026+f)
    gb.fit(lgb_frame(a),y[tr],eval_set=[(lgb_frame(b),y[va])],callbacks=[lgb.early_stopping(150,verbose=False)])
    oof['lgb'][va]=gb.predict_proba(lgb_frame(b))[:,1];pred['lgb']+=gb.predict_proba(lgb_frame(c))[:,1]/5
    for k in oof:rows.append({'fold':f,'model':k,'accuracy':accuracy_score(y[va],oof[k][va]>.5),'auc':roc_auc_score(y[va],oof[k][va]),'logloss':log_loss(y[va],oof[k][va])})
    importances.append(pd.Series(cat.feature_importances_,index=X.columns))
    print(rows[-2:])
pd.DataFrame(rows).to_csv(W/'fold_metrics.csv',index=False)
for k in oof:print(k,'OOF accuracy',accuracy_score(y,oof[k]>.5),'AUC',roc_auc_score(y,oof[k]))


# Copyright 2026 Parth Maniar. Apache-2.0.
choices=[]
for weight in [0,.25,.5,.75,1]:
    p=weight*oof['cat']+(1-weight)*oof['lgb']
    for threshold in [.45,.475,.5,.525,.55]:choices.append({'cat_weight':weight,'threshold':threshold,'accuracy':accuracy_score(y,p>threshold)})
search=pd.DataFrame(choices).sort_values(['accuracy','threshold'],ascending=[False,True]);search.to_csv(W/'blend_search.csv',index=False)
best=search.iloc[0];weight=float(best.cat_weight);threshold=float(best.threshold)
blend=weight*oof['cat']+(1-weight)*oof['lgb']; pt=weight*pred['cat']+(1-weight)*pred['lgb']
print('Selected development setting',best.to_dict())
print('Fixed equal blend at .5',accuracy_score(y,(oof['cat']+oof['lgb'])/2>.5))
pd.concat(importances,axis=1).mean(axis=1).sort_values().tail(15).plot.barh(figsize=(7,5),title='CatBoost mean fold importance');plt.tight_layout();plt.savefig(W/'feature_importance.png');plt.show()
plt.imshow(confusion_matrix(y,blend>threshold));plt.title('OOF confusion matrix, selected threshold');plt.colorbar();plt.tight_layout();plt.savefig(W/'confusion_matrix.png');plt.show()


# Copyright 2026 Parth Maniar. Apache-2.0.
group_oof=np.zeros(len(X));audit=[]
for f,(tr,va) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=2026).split(X,y,groups)):
    assert set(groups.iloc[tr]).isdisjoint(groups.iloc[va])
    a,b,_=fold_frames(X.iloc[tr].copy(),X.iloc[va].copy(),XT.copy())
    cat=CatBoostClassifier(iterations=1800,depth=6,learning_rate=.035,l2_leaf_reg=6,loss_function='Logloss',random_seed=2026+f,thread_count=4,verbose=False,allow_writing_files=False)
    cat.fit(a,y[tr],cat_features=CATS,eval_set=(b,y[va]),early_stopping_rounds=150)
    group_oof[va]=cat.predict_proba(b)[:,1]
    audit.append({'fold':f,'accuracy':accuracy_score(y[va],group_oof[va]>.5),'validation_rows':len(va)})
pd.DataFrame(audit).to_csv(W/'group_audit.csv',index=False)
print('Group-disjoint CatBoost OOF accuracy',accuracy_score(y,group_oof>.5))
print('Random-split CatBoost OOF accuracy',accuracy_score(y,oof['cat']>.5))


# Copyright 2026 Parth Maniar. Apache-2.0.
assert np.isfinite(pt).all() and ((pt>=0)&(pt<=1)).all()
sub=sample.copy();sub['Transported']=pt>threshold
assert sub.PassengerId.tolist()==sample.PassengerId.tolist() and len(sub)==len(test)
sub.to_csv(W/'submission.csv',index=False)
pd.DataFrame({'PassengerId':train.PassengerId,'target':y,'cat':oof['cat'],'lgb':oof['lgb'],'blend':blend,'group_audit_cat':group_oof}).to_csv(W/'oof_predictions.csv',index=False)
pd.DataFrame({'PassengerId':test.PassengerId,'cat':pred['cat'],'lgb':pred['lgb'],'blend':pt}).to_csv(W/'test_probabilities.csv',index=False)
metrics={'cat_oof_accuracy':accuracy_score(y,oof['cat']>.5),'lgb_oof_accuracy':accuracy_score(y,oof['lgb']>.5),'selected_blend_oof_accuracy':float(best.accuracy),'cat_weight':weight,'threshold':threshold,'group_disjoint_cat_accuracy':accuracy_score(y,group_oof>.5),'runtime_seconds':time.monotonic()-start,'caveat':'Threshold/blend selection is development tuning, not unbiased validation.'}
(W/'metrics.json').write_text(json.dumps(metrics,indent=2))
manifest=[{'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(W.iterdir()) if p.is_file()]
(W/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(metrics);print('Output IDs checked:',len(sub))
