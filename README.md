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
