#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval', 'chat_benchmarks', 'AMC23'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval', 'chat_benchmarks', 'AIME24'))

from eval.chat_benchmarks.AMC23.eval_instruct import AMC23Benchmark
from eval.chat_benchmarks.AIME24.eval_instruct import AIME24Benchmark

# 테스트용 모델 출력 샘플들
test_outputs = [
    """<think>
Okay, let's see. So there's this problem about Alicia and Beth biking towards each other from two cities, A and B, which are 45 miles apart. Alicia starts from city A and bikes towards B at 18 mph, while Beth starts from city B at the same time and bikes towards A at 12 mph. The question is asking how many miles from city A they will be when they meet. Hmm, okay.

Alright, let me think. This seems like a relative speed problem. When two objects move towards each other, their relative speed is the sum of their individual speeds. So, Alicia is going at 18 mph, and Beth is going at 12 mph. So combined, they're approaching each other at 18 + 12 = 30 mph. Right?

So if they're moving towards each other with a combined speed of 30 mph, and the distance between them is 45 miles,""",
    
    """Looking at this problem step by step:
First, I need to find when they meet. Since they're moving towards each other, their relative speed is 18 + 12 = 30 mph.
Time to meet = 45 miles ÷ 30 mph = 1.5 hours
Distance from city A = 18 mph × 1.5 hours = 27 miles
Therefore, the answer is 27.""",
    
    """Let me solve this step by step.
The combined speed is 18 + 12 = 30 mph.
They will meet in 45/30 = 1.5 hours.
Alicia travels 18 × 1.5 = 27 miles from city A.
The answer is 27""",
    
    """This is a classic relative motion problem.
Combined approach speed: 18 + 12 = 30 mph
Time until meeting: 45/30 = 1.5 hours  
Distance from A: 18 × 1.5 = 27 miles
Final answer: 27""",
    
    """\\boxed{27}""",
    
    """The solution is \\boxed{27}."""
]

def test_answer_extraction_benchmark(benchmark_class, benchmark_name):
    benchmark = benchmark_class()
    
    print(f"Testing {benchmark_name} answer extraction with improved logic:")
    print("=" * 60)
    
    # Test with expected answers
    expected_answers = ["27", "27", "27", "27", "27", "27"]
    correct_extractions = 0
    
    for i, (output, expected) in enumerate(zip(test_outputs, expected_answers)):
        extracted = benchmark.extract_answer(output)
        if extracted == expected:
            correct_extractions += 1
            print(f"✅ Test {i+1}: PASS (extracted: '{extracted}')")
        else:
            print(f"❌ Test {i+1}: FAIL (extracted: '{extracted}', expected: '{expected}')")
            print(f"   Output sample: {output[:100]}...")
    
    print(f"\n{benchmark_name} Results: {correct_extractions}/{len(test_outputs)} tests passed")
    return correct_extractions == len(test_outputs)

def test_all_benchmarks():
    print("Testing answer extraction for math benchmarks")
    print("=" * 60)
    
    benchmarks = [
        (AMC23Benchmark, "AMC23"),
        (AIME24Benchmark, "AIME24")
    ]
    
    all_passed = True
    
    for benchmark_class, benchmark_name in benchmarks:
        try:
            passed = test_answer_extraction_benchmark(benchmark_class, benchmark_name)
            all_passed = all_passed and passed
            print()
        except Exception as e:
            print(f"❌ Error testing {benchmark_name}: {e}")
            all_passed = False
            print()
    
    print("=" * 60)
    if all_passed:
        print("🎉 All benchmarks passed!")
    else:
        print("⚠️  Some benchmarks failed. Check the output above.")
    
    return all_passed

if __name__ == "__main__":
    success = test_all_benchmarks()
    sys.exit(0 if success else 1) 