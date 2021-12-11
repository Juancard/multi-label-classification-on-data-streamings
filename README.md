# multi-label-classification-on-data-streamings

```bat
git submodule update --init --recursive
cd scikit-multiflow/
pip install -r requirements-dev.txt
pip install -r requirements.txt
python setup.py test
cd ../
pip install -r requirements.txt
ln -s scikit-multiflow/src/skmultiflow/ skmultiflow
```

Run test on file:
```bash
PYTHONPATH=./multi-label-classification-on-data-streamings pytest scikit-multiflow/tests/meta/test_dynamic_weighted_majority.py --showlocals -v
```
