#!/bin/sh

cd action_robust
python parse_arrl.py
cp action_robust*.json ..

cd ../adversarial_rl
python parse_adversarial_rl.py
cp adversarial_robust*.json ..

cd domain_randomization
python parse_dr.py
cp parameter_robust*.json ..
