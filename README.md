# simple_adversarial_generator
Simple adversarial input generator for VNN-COMP using random inputs

Example usage:

```python3 randgen.py test_unsat.onnx test_prop.vnnlib out.txt```
should output "unknown" to out.txt


```python3 randgen.py test_sat.onnx test_prop.vnnlib out.txt```
should output "sat" to out.txt
