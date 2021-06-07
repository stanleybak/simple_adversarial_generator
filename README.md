[![Build Status](https://travis-ci.com/stanleybak/simple_adversarial_generator.svg?branch=main)](https://travis-ci.com/stanleybak/simple_adversarial_generator)

# simple_adversarial_generator
Simple adversarial input generator for VNN-COMP using random inputs. In the future we may add true adversarial example generation like PGD.

Example usage:

```python3 randgen.py test_unsat.onnx test_prop.vnnlib out.txt```
should output "unknown" to out.txt

```python3 randgen.py test_sat.onnx test_prop.vnnlib out.txt```
should output "sat" to out.txt
