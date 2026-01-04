#!/usr/bin/env python3
"""
Het-Benchmark Knowledge Graph Query Example

This script demonstrates how to query the MOH-KG knowledge graph
for various insights about models, operators, and hardware platforms.
"""

import sys
import json
sys.path.insert(0, '..')


def main():
    print("="*60)
    print("MOH-KG Knowledge Graph Query Example")
    print("="*60)
    
    # Load knowledge graph
    print("\n[1] Loading Knowledge Graph...")
    with open('../data/moh_kg.json', 'r') as f:
        kg = json.load(f)
    
    nodes = kg['nodes']
    edges = kg['edges']
    stats = kg['statistics']
    
    print(f"    Total nodes: {stats['total_nodes']}")
    print(f"    Total edges: {stats['total_edges']}")
    
    # Query 1: Get all hardware platforms
    print("\n[2] Query: Hardware Platforms")
    print("-"*40)
    hw_nodes = [n for n in nodes if n['node_type'] == 'hardware']
    for hw in hw_nodes:
        props = hw['properties']
        print(f"    {props.get('platform_name', hw['node_id'])}")
        print(f"      Vendor: {props.get('vendor')}")
        print(f"      Peak FLOPS: {props.get('peak_flops_tflops')} TFLOPS")
        print(f"      Memory BW: {props.get('memory_bandwidth_gbps')} GB/s")
    
    # Query 2: Get operator types
    print("\n[3] Query: Operator Types")
    print("-"*40)
    op_type_nodes = [n for n in nodes if n['node_type'] == 'operator_type']
    print(f"    Total operator types: {len(op_type_nodes)}")
    for op in op_type_nodes[:8]:
        props = op['properties']
        print(f"    - {props.get('type_name', op['node_id'])}: {props.get('category', 'N/A')}")
    if len(op_type_nodes) > 8:
        print(f"    ... and {len(op_type_nodes)-8} more")
    
    # Query 3: Get models
    print("\n[4] Query: Models")
    print("-"*40)
    model_nodes = [n for n in nodes if n['node_type'] == 'model']
    print(f"    Total models: {len(model_nodes)}")
    
    # Group by category
    categories = {}
    for m in model_nodes:
        cat = m['properties'].get('category', 'Unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items()):
        print(f"    - {cat}: {count} models")
    
    # Query 4: Edge type distribution
    print("\n[5] Query: Edge Types")
    print("-"*40)
    edge_types = stats['edge_types']
    for edge_type, count in sorted(edge_types.items(), key=lambda x: -x[1]):
        print(f"    {edge_type}: {count:,}")
    
    # Query 5: Find operators for a specific model
    print("\n[6] Query: Operators in Qwen2.5-7B")
    print("-"*40)
    
    # Find model node
    model_id = None
    for n in model_nodes:
        if 'Qwen' in n['properties'].get('model_name', ''):
            model_id = n['node_id']
            break
    
    if model_id:
        # Find r_contains edges from this model
        contains_edges = [e for e in edges if e['relation'] == 'r_contains' and e['source'] == model_id]
        print(f"    Model: {model_id}")
        print(f"    Operators: {len(contains_edges)}")
        
        # Get operator types
        op_types = {}
        for edge in contains_edges:
            op_id = edge['target']
            op_node = next((n for n in nodes if n['node_id'] == op_id), None)
            if op_node:
                op_type = op_node['properties'].get('operator_type', 'Unknown')
                op_types[op_type] = op_types.get(op_type, 0) + 1
        
        print("    Operator distribution:")
        for op_type, count in sorted(op_types.items(), key=lambda x: -x[1])[:5]:
            print(f"      - {op_type}: {count}")
    
    # Query 6: Hardware support for operator types
    print("\n[7] Query: Hardware Support for Linear")
    print("-"*40)
    
    # Find Linear operator type
    linear_type_id = None
    for n in op_type_nodes:
        if n['properties'].get('type_name') == 'Linear':
            linear_type_id = n['node_id']
            break
    
    if linear_type_id:
        # Find r_supports edges to Linear
        supports_edges = [e for e in edges if e['relation'] == 'r_supports' and e['target'] == linear_type_id]
        print(f"    Operator: Linear")
        print(f"    Supported by {len(supports_edges)} platforms:")
        
        for edge in supports_edges:
            hw_id = edge['source']
            hw_node = next((n for n in hw_nodes if n['node_id'] == hw_id), None)
            if hw_node:
                props = hw_node['properties']
                support_level = edge.get('properties', {}).get('support_level', 1.0)
                print(f"      - {props.get('platform_name')}: {support_level:.2f}")
    
    print("\n" + "="*60)
    print("Knowledge Graph Query Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
