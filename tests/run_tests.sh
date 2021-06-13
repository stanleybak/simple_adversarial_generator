#!/bin/bash -e
 
python3 ../src/agen/randgen.py test_sat.onnx test_prop.vnnlib out.txt
grep "violated" out.txt

python3 ../src/agen/randgen.py test_unsat.onnx test_prop.vnnlib out.txt
grep "unknown" out.txt

# parsing tests for disjunctions
python3 ../src/agen/randgen.py test_unsat.onnx prop_5.vnnlib out.txt

python3 ../src/agen/randgen.py test_unsat.onnx prop_6.vnnlib out.txt

python3 ../src/agen/randgen.py test_unsat.onnx prop_7.vnnlib out.txt

echo "Tests Passed"
