
## To run tests

```bash
cd code
```

Run all tests  
```bash
python -m unittest
```

Run single tests
```bash
python -m unittest tests.test_manage
```

## Requirements
Note: Schmugge dataset, and light and medium models are needed to run tests

Also initialize their csv files:  
```bash
python main.py reset -d Schmugge -p
python main.py reset -d light -p
python main.py reset -d medium -p
```
