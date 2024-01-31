# Raven ML Dataset Plugins

## Installation

1. Follow the [instructions to setup ravenML](https://github.com/autognc/ravenML). (Change python version to 3.7.0)

2. Activate the ravenML environment: 
```bash
conda activate ravenml
```

3. Move into the proper directory: 
```bash
cd rmldatatfrecord
```

4. Install requirements: 
```bash
pip install -e .
```

## Dataset Creation

1. Move to the proper directory: 
```bash
cd rmldatatfrecord
```

2. Make a copy of the config template: 
```bash
cp sample_configs/tf_record_all_fields.yaml config.yaml
```

3. Modify the 'imageset' parameter to point to the local path for the dataset.

4. Create the dataset: 
```bash
ravenml data create -c config.yaml tf-record
```

