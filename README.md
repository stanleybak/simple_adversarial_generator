# simple_adversarial_generator
Simple adversarial input generator for VNN-COMP

consists of two methods to try to generate safety property violations:

1. randgen - randomly tries inputs to see if they violate the property

2. agen - converts onnx network to tensorflow1 and uses the foolbox library to try to generate adversarial images
