cp *.py ../
cd ..
python3 test_gen.py
python3 test_model_names.py
python3 test_save.py
python3 test_wmode.py
rm -rf test_*.py tests/tests_temp