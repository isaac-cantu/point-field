import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from notebook_tools import modelnet_manifest, freeze_splits, exact_d2, stable_seed
from pointfield.learning import ChristoffelDarboux, MultiViewMetric, knn_density_weights
from pointfield.geometry.mesh import TriangleMesh

@pytest.mark.parametrize('reg',[-1,np.nan,np.inf])
def test_invalid_ridge(reg):
    with pytest.raises(ValueError):ChristoffelDarboux(reg=reg)

def test_invalid_query_and_weights():
    X=np.random.default_rng(0).normal(size=(20,2))
    cd=ChristoffelDarboux().fit(X)
    with pytest.raises(ValueError):cd.score(np.ones((3,3)))
    with pytest.raises(ValueError):cd.fit(X,np.full(20,np.nan))

def test_multiview_no_variation():
    metric=MultiViewMetric().fit(np.ones((6,3,5)))
    assert np.isfinite(metric.W_).all()
    np.testing.assert_allclose(metric.W_,np.eye(5))

def test_multiview_small():
    with pytest.raises(ValueError):MultiViewMetric().fit(np.ones((2,3,4)))

def test_duplicate_weights():
    with pytest.raises(ValueError):knn_density_weights(np.ones((30,3)))

def test_mesh_negative_indices():
    with pytest.raises(ValueError):TriangleMesh(np.eye(3),np.array([[-1,0,1]]))

def test_manifest_cross_split_duplicates(tmp_path):
    text='OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n'
    for split,name,suffix in [('train','a',''),('train','b','\n'),('test','duplicate',''),('test','c','\n\n')]:
        p=tmp_path/'class'/split/f'{name}.off';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text+suffix)
    df,rejected=modelnet_manifest(tmp_path,1)
    assert len(rejected)==1 and rejected.official_split.iloc[0]=='test'
    frozen=freeze_splits(df,1,(1,1,1))
    assert frozen.sha256.nunique()==3
    with pytest.raises(ValueError):freeze_splits(df,1,(2,1,1))

def test_d2_permutation():
    X=np.random.default_rng(0).normal(size=(45,3))
    np.testing.assert_allclose(exact_d2(X),exact_d2(X[::-1]))
    assert stable_seed(1,'object')==stable_seed(1,'object')

def test_modelnet_pipeline_with_generated_off_fixtures(tmp_path):
    # Synthetic OFF fixtures test the adapter, NOT ModelNet performance.
    from notebook_tools import load_partition, fit_models, evaluate
    for cls in range(10):
        for split,count in [('train',3),('test',1)]:
            for i in range(count):
                z=1+.03*cls+.001*i+(.0001 if split=='test' else 0)
                p=tmp_path/f'c{cls}'/split/f'{i}.off';p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text(f'OFF4 4 0\n0 0 0\n1 0 0\n0 1 0\n0 0 {z}\n3 0 1 2\n3 0 1 3\n3 0 2 3\n3 1 2 3\n')
    df,_=modelnet_manifest(tmp_path,10)
    manifest=freeze_splits(df,2,(2,1,1))
    R,yr,_=load_partition(manifest,'reference',32)
    Q,yq,ids=load_partition(manifest,'test',32)
    models,_=fit_models(R,2,n_views=2)
    summary,individual,ci,_=evaluate(models,Q,yq,yr,ids,n_boot=10,cluster_seeds=(0,))
    assert len(summary)==7 and len(individual)==70
    assert np.isfinite(summary[['mAP','nn','ARI']].to_numpy()).all()

def test_cross_label_quarantine(tmp_path):
    for c in ('a','b'):
        for split in ('train','test'):
            d=tmp_path/c/split; d.mkdir(parents=True)
            (d/'valid.off').write_text(c+split)
            (d/'ambiguous.off').write_text('same bytes in both classes')
    df,out=modelnet_manifest(tmp_path,2)
    assert len(df)==4 and len(out)==4
    assert set(out.exclusion_reason)=={'conflicting_class_labels'}
    assert not set(out.sha256)&set(df.sha256)
