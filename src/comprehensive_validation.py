#!/usr/bin/env python3
"""
Comprehensive Validation Script for Het-Benchmark
Checks all components for completeness and correctness
"""

import json
import os
import sys
from typing import Dict, List, Tuple

class ValidationResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
    
    def add_pass(self, check: str, details: str = ""):
        self.passed.append((check, details))
    
    def add_fail(self, check: str, details: str = ""):
        self.failed.append((check, details))
    
    def add_warning(self, check: str, details: str = ""):
        self.warnings.append((check, details))
    
    def print_report(self):
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)
        
        print(f"\n✅ PASSED ({len(self.passed)}):")
        for check, details in self.passed:
            print(f"   • {check}")
            if details:
                print(f"     {details}")
        
        if self.warnings:
            print(f"\n⚠️ WARNINGS ({len(self.warnings)}):")
            for check, details in self.warnings:
                print(f"   • {check}")
                if details:
                    print(f"     {details}")
        
        if self.failed:
            print(f"\n❌ FAILED ({len(self.failed)}):")
            for check, details in self.failed:
                print(f"   • {check}")
                if details:
                    print(f"     {details}")
        
        print("\n" + "="*60)
        if self.failed:
            print(f"RESULT: FAILED - {len(self.failed)} issues need to be fixed")
        elif self.warnings:
            print(f"RESULT: PASSED WITH WARNINGS - {len(self.warnings)} warnings")
        else:
            print("RESULT: ALL CHECKS PASSED ✅")
        print("="*60)
        
        return len(self.failed) == 0

def validate_model_dataset(result: ValidationResult) -> Dict:
    """Validate model_dataset.json"""
    print("\n[1/6] Validating Model Dataset...")
    
    try:
        with open('data/model_dataset.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        result.add_fail("Model dataset file", f"Cannot load: {e}")
        return None
    
    # Check metadata
    if 'metadata' not in data:
        result.add_fail("Model dataset metadata", "Missing metadata section")
    else:
        result.add_pass("Model dataset metadata", f"Version: {data['metadata'].get('version', 'N/A')}")
    
    # Check model count
    models = data.get('models', [])
    if len(models) < 34:
        result.add_fail("Model count", f"Only {len(models)} models, need at least 34")
    else:
        result.add_pass("Model count", f"{len(models)} models")
    
    # Check each model has operators
    models_without_ops = []
    total_ops = 0
    for model in models:
        ops = model.get('operators', [])
        if len(ops) == 0:
            models_without_ops.append(model.get('name', 'unknown'))
        total_ops += len(ops)
    
    if models_without_ops:
        result.add_fail("Model operators", f"{len(models_without_ops)} models without operators: {models_without_ops[:5]}...")
    else:
        result.add_pass("Model operators", f"All models have operators, total: {total_ops}")
    
    # Check operator types
    op_types = set()
    for model in models:
        for op in model.get('operators', []):
            op_types.add(op.get('type', 'unknown'))
    
    if len(op_types) < 10:
        result.add_warning("Operator types", f"Only {len(op_types)} types, expected more variety")
    else:
        result.add_pass("Operator types", f"{len(op_types)} unique types")
    
    # Check categories
    categories = {}
    for model in models:
        cat = model.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    required_cats = ['LLM', 'CV', 'NLP']
    missing_cats = [c for c in required_cats if c not in categories]
    if missing_cats:
        result.add_fail("Model categories", f"Missing categories: {missing_cats}")
    else:
        result.add_pass("Model categories", f"{len(categories)} categories: {dict(categories)}")
    
    return data

def validate_knowledge_graph(result: ValidationResult) -> Dict:
    """Validate moh_kg.json"""
    print("\n[2/6] Validating Knowledge Graph...")
    
    try:
        with open('data/moh_kg.json', 'r') as f:
            kg = json.load(f)
    except Exception as e:
        result.add_fail("Knowledge graph file", f"Cannot load: {e}")
        return None
    
    # Check node count
    nodes = kg.get('nodes', [])
    if len(nodes) < 1000:
        result.add_fail("KG node count", f"Only {len(nodes)} nodes, expected more")
    else:
        result.add_pass("KG node count", f"{len(nodes)} nodes")
    
    # Check edge count
    edges = kg.get('edges', [])
    if len(edges) < 5000:
        result.add_fail("KG edge count", f"Only {len(edges)} edges, expected more")
    else:
        result.add_pass("KG edge count", f"{len(edges)} edges")
    
    # Check edge types
    edge_types = {}
    for edge in edges:
        et = edge.get('edge_type', 'unknown')
        edge_types[et] = edge_types.get(et, 0) + 1
    
    required_edge_types = ['r_contains', 'r_has_type', 'r_supports', 'r_seq', 'r_sim', 'r_perf']
    missing_types = [t for t in required_edge_types if t not in edge_types]
    
    if missing_types:
        result.add_fail("KG edge types", f"Missing edge types: {missing_types}")
    else:
        result.add_pass("KG edge types", f"All 6 edge types present: {dict(edge_types)}")
    
    # Check node types
    node_types = {}
    for node in nodes:
        nt = node.get('node_type', 'unknown')
        node_types[nt] = node_types.get(nt, 0) + 1
    
    required_node_types = ['hardware', 'operator_type', 'model', 'operator_instance']
    missing_node_types = [t for t in required_node_types if t not in node_types]
    
    if missing_node_types:
        result.add_fail("KG node types", f"Missing node types: {missing_node_types}")
    else:
        result.add_pass("KG node types", f"All 4 node types present: {dict(node_types)}")
    
    return kg

def validate_standard_inputs(result: ValidationResult) -> Dict:
    """Validate standard_inputs.json"""
    print("\n[3/6] Validating Standard Inputs...")
    
    try:
        with open('data/standard_inputs.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        result.add_fail("Standard inputs file", f"Cannot load: {e}")
        return None
    
    # Count total inputs
    total = 0
    categories = {}
    for key in ['llm_inputs', 'cv_inputs', 'vlm_inputs', 'diffusion_inputs']:
        if key in data:
            count = len(data[key])
            categories[key] = count
            total += count
    
    if total < 1000:
        result.add_fail("Standard input count", f"Only {total} inputs, need at least 1000")
    else:
        result.add_pass("Standard input count", f"{total} inputs")
    
    # Check category distribution
    expected = {'llm_inputs': 300, 'cv_inputs': 250, 'vlm_inputs': 200, 'diffusion_inputs': 250}
    for cat, expected_count in expected.items():
        actual = categories.get(cat, 0)
        if actual < expected_count:
            result.add_warning(f"Standard inputs {cat}", f"Only {actual}, expected {expected_count}")
        else:
            result.add_pass(f"Standard inputs {cat}", f"{actual} inputs")
    
    return data

def validate_source_code(result: ValidationResult):
    """Validate source code files"""
    print("\n[4/6] Validating Source Code...")
    
    required_files = [
        ('src/hal.py', 'Hardware Abstraction Layer'),
        ('src/model_parser.py', 'Model Parser'),
        ('src/operator_extractor.py', 'Operator Extractor'),
        ('src/copa.py', 'COPA Algorithm'),
        ('src/moh_kg.py', 'MOH-KG'),
        ('src/rgat.py', 'RGAT Model'),
        ('src/kg_a2o.py', 'KG-A2O Algorithm'),
        ('src/profiler.py', 'Performance Profiler'),
    ]
    
    for filepath, name in required_files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            if size < 1000:
                result.add_warning(f"Source: {name}", f"File too small ({size} bytes)")
            else:
                result.add_pass(f"Source: {name}", f"{size} bytes")
        else:
            result.add_fail(f"Source: {name}", f"File not found: {filepath}")

def validate_experiment_results(result: ValidationResult):
    """Validate experiment result files"""
    print("\n[5/6] Validating Experiment Results...")
    
    required_tables = [
        ('results/table4_model_dataset.csv', 'Table 4: Model Dataset', 30),
        ('results/table5_operator_coverage.csv', 'Table 5: Operator Coverage', 5),
        ('results/table6_performance_profiling.csv', 'Table 6: Performance Profiling', 10),
        ('results/table7_copa_attribution.csv', 'Table 7: COPA Attribution', 5),
        ('results/table8_cross_platform_prediction.csv', 'Table 8: Cross-Platform Prediction', 20),
    ]
    
    for filepath, name, min_rows in required_tables:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                lines = f.readlines()
            row_count = len(lines) - 1  # Exclude header
            if row_count < min_rows:
                result.add_warning(name, f"Only {row_count} rows, expected at least {min_rows}")
            else:
                result.add_pass(name, f"{row_count} rows")
        else:
            result.add_fail(name, f"File not found: {filepath}")

def validate_trained_model(result: ValidationResult):
    """Validate trained RGAT model"""
    print("\n[6/6] Validating Trained Model...")
    
    model_path = 'models/rgat_final.pt'
    report_path = 'models/training_report.json'
    
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        if size < 100000:  # Less than 100KB
            result.add_warning("RGAT model file", f"File too small ({size} bytes)")
        else:
            result.add_pass("RGAT model file", f"{size/1024/1024:.2f} MB")
    else:
        result.add_fail("RGAT model file", "File not found")
    
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report = json.load(f)
        epochs = report.get('epochs', 0)
        loss = report.get('best_loss', float('inf'))
        if epochs < 50:
            result.add_warning("RGAT training", f"Only {epochs} epochs")
        else:
            result.add_pass("RGAT training", f"{epochs} epochs, best loss: {loss:.4f}")
    else:
        result.add_fail("RGAT training report", "File not found")

def main():
    print("="*60)
    print("HET-BENCHMARK COMPREHENSIVE VALIDATION")
    print("="*60)
    
    os.chdir('/home/ubuntu/het-benchmark')
    
    result = ValidationResult()
    
    # Run all validations
    model_data = validate_model_dataset(result)
    kg_data = validate_knowledge_graph(result)
    input_data = validate_standard_inputs(result)
    validate_source_code(result)
    validate_experiment_results(result)
    validate_trained_model(result)
    
    # Print report
    success = result.print_report()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
