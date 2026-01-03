"""
MOH-KG: Model-Operator-Hardware Knowledge Graph
Multi-relational knowledge graph for AI model migration evaluation
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from enum import Enum
from collections import defaultdict
import hashlib
from loguru import logger


class EntityType(Enum):
    """Entity types in MOH-KG"""
    MODEL = "Model"
    OPERATOR = "Operator"
    HARDWARE = "Hardware"
    OPERATOR_INSTANCE = "OperatorInstance"
    PERFORMANCE_RECORD = "PerformanceRecord"


class RelationType(Enum):
    """Relation types in MOH-KG"""
    # Model-Operator relations
    CONTAINS = "contains"           # Model contains Operator
    DEPENDS_ON = "depends_on"       # Operator depends on another Operator
    
    # Operator-Hardware relations
    RUNS_ON = "runs_on"             # Operator runs on Hardware
    SUPPORTED_BY = "supported_by"   # Operator is supported by Hardware
    OPTIMIZED_FOR = "optimized_for" # Operator is optimized for Hardware
    
    # Performance relations
    HAS_PERFORMANCE = "has_performance"  # Entity has performance record
    COMPARED_TO = "compared_to"          # Performance comparison
    
    # Similarity relations
    SIMILAR_TO = "similar_to"       # Semantic similarity between operators
    EQUIVALENT_TO = "equivalent_to" # Functionally equivalent operators


@dataclass
class Entity:
    """Base entity in knowledge graph"""
    id: str
    type: EntityType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "properties": self.properties,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Entity":
        return cls(
            id=data["id"],
            type=EntityType(data["type"]),
            name=data["name"],
            properties=data.get("properties", {}),
        )


@dataclass
class Relation:
    """Relation between entities"""
    id: str
    type: RelationType
    source_id: str
    target_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "properties": self.properties,
            "weight": self.weight,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Relation":
        return cls(
            id=data["id"],
            type=RelationType(data["type"]),
            source_id=data["source_id"],
            target_id=data["target_id"],
            properties=data.get("properties", {}),
            weight=data.get("weight", 1.0),
        )


@dataclass
class ModelEntity(Entity):
    """Model entity with specific properties"""
    def __init__(
        self,
        model_id: str,
        name: str,
        category: str,
        num_params: int,
        architecture: str,
        **kwargs
    ):
        super().__init__(
            id=model_id,
            type=EntityType.MODEL,
            name=name,
            properties={
                "category": category,
                "num_params": num_params,
                "architecture": architecture,
                **kwargs,
            }
        )


@dataclass
class OperatorEntity(Entity):
    """Operator entity with specific properties"""
    def __init__(
        self,
        operator_id: str,
        name: str,
        op_type: str,
        category: str,
        **kwargs
    ):
        super().__init__(
            id=operator_id,
            type=EntityType.OPERATOR,
            name=name,
            properties={
                "op_type": op_type,
                "category": category,
                **kwargs,
            }
        )


@dataclass
class HardwareEntity(Entity):
    """Hardware entity with specific properties"""
    def __init__(
        self,
        hardware_id: str,
        name: str,
        platform: str,
        vendor: str,
        compute_capability: str,
        memory_gb: float,
        **kwargs
    ):
        super().__init__(
            id=hardware_id,
            type=EntityType.HARDWARE,
            name=name,
            properties={
                "platform": platform,
                "vendor": vendor,
                "compute_capability": compute_capability,
                "memory_gb": memory_gb,
                **kwargs,
            }
        )


class MOHKG:
    """
    Model-Operator-Hardware Knowledge Graph
    
    A multi-relational knowledge graph that captures:
    - Model structure and operator composition
    - Operator characteristics and dependencies
    - Hardware capabilities and operator support
    - Cross-platform performance relationships
    """
    
    def __init__(self):
        self._entities: Dict[str, Entity] = {}
        self._relations: Dict[str, Relation] = {}
        
        # Indexes for efficient querying
        self._entity_by_type: Dict[EntityType, Set[str]] = defaultdict(set)
        self._relations_by_source: Dict[str, Set[str]] = defaultdict(set)
        self._relations_by_target: Dict[str, Set[str]] = defaultdict(set)
        self._relations_by_type: Dict[RelationType, Set[str]] = defaultdict(set)
        
        self._relation_counter = 0
    
    def add_entity(self, entity: Entity) -> str:
        """Add an entity to the knowledge graph"""
        if entity.id in self._entities:
            logger.warning(f"Entity {entity.id} already exists, updating...")
        
        self._entities[entity.id] = entity
        self._entity_by_type[entity.type].add(entity.id)
        
        return entity.id
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        properties: Optional[Dict] = None,
        weight: float = 1.0
    ) -> str:
        """Add a relation between entities"""
        if source_id not in self._entities:
            raise ValueError(f"Source entity {source_id} not found")
        if target_id not in self._entities:
            raise ValueError(f"Target entity {target_id} not found")
        
        self._relation_counter += 1
        relation_id = f"rel_{self._relation_counter:06d}"
        
        relation = Relation(
            id=relation_id,
            type=relation_type,
            source_id=source_id,
            target_id=target_id,
            properties=properties or {},
            weight=weight,
        )
        
        self._relations[relation_id] = relation
        self._relations_by_source[source_id].add(relation_id)
        self._relations_by_target[target_id].add(relation_id)
        self._relations_by_type[relation_type].add(relation_id)
        
        return relation_id
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID"""
        return self._entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a specific type"""
        return [self._entities[eid] for eid in self._entity_by_type[entity_type]]
    
    def get_relations(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relation_type: Optional[RelationType] = None
    ) -> List[Relation]:
        """Query relations with optional filters"""
        result_ids = None
        
        if source_id is not None:
            result_ids = self._relations_by_source.get(source_id, set())
        
        if target_id is not None:
            target_ids = self._relations_by_target.get(target_id, set())
            if result_ids is None:
                result_ids = target_ids
            else:
                result_ids = result_ids & target_ids
        
        if relation_type is not None:
            type_ids = self._relations_by_type.get(relation_type, set())
            if result_ids is None:
                result_ids = type_ids
            else:
                result_ids = result_ids & type_ids
        
        if result_ids is None:
            result_ids = set(self._relations.keys())
        
        return [self._relations[rid] for rid in result_ids]
    
    def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None,
        direction: str = "outgoing"
    ) -> List[Tuple[Entity, Relation]]:
        """Get neighboring entities connected by relations"""
        neighbors = []
        
        if direction in ["outgoing", "both"]:
            for rel_id in self._relations_by_source.get(entity_id, set()):
                rel = self._relations[rel_id]
                if relation_type is None or rel.type == relation_type:
                    neighbor = self._entities.get(rel.target_id)
                    if neighbor:
                        neighbors.append((neighbor, rel))
        
        if direction in ["incoming", "both"]:
            for rel_id in self._relations_by_target.get(entity_id, set()):
                rel = self._relations[rel_id]
                if relation_type is None or rel.type == relation_type:
                    neighbor = self._entities.get(rel.source_id)
                    if neighbor:
                        neighbors.append((neighbor, rel))
        
        return neighbors
    
    def get_model_operators(self, model_id: str) -> List[Entity]:
        """Get all operators contained in a model"""
        operators = []
        for neighbor, rel in self.get_neighbors(model_id, RelationType.CONTAINS):
            if neighbor.type == EntityType.OPERATOR:
                operators.append(neighbor)
        return operators
    
    def get_operator_hardware_support(self, operator_id: str) -> List[Entity]:
        """Get all hardware platforms that support an operator"""
        hardware_list = []
        for neighbor, rel in self.get_neighbors(operator_id, RelationType.SUPPORTED_BY):
            if neighbor.type == EntityType.HARDWARE:
                hardware_list.append(neighbor)
        return hardware_list
    
    def get_hardware_operators(self, hardware_id: str) -> List[Entity]:
        """Get all operators supported by a hardware platform"""
        operators = []
        for neighbor, rel in self.get_neighbors(hardware_id, direction="incoming"):
            if rel.type == RelationType.SUPPORTED_BY:
                operators.append(neighbor)
        return operators
    
    def calculate_migration_compatibility(
        self,
        model_id: str,
        source_hardware_id: str,
        target_hardware_id: str
    ) -> Dict[str, Any]:
        """
        Calculate migration compatibility score between hardware platforms
        
        Returns:
            Dictionary with compatibility metrics
        """
        model_operators = self.get_model_operators(model_id)
        
        source_supported = set()
        target_supported = set()
        
        for op in model_operators:
            source_hw = self.get_operator_hardware_support(op.id)
            target_hw = self.get_operator_hardware_support(op.id)
            
            if any(h.id == source_hardware_id for h in source_hw):
                source_supported.add(op.id)
            if any(h.id == target_hardware_id for h in target_hw):
                target_supported.add(op.id)
        
        total_ops = len(model_operators)
        source_coverage = len(source_supported) / total_ops if total_ops > 0 else 0
        target_coverage = len(target_supported) / total_ops if total_ops > 0 else 0
        
        # Operators that need migration attention
        migration_gaps = source_supported - target_supported
        
        return {
            "model_id": model_id,
            "source_hardware": source_hardware_id,
            "target_hardware": target_hardware_id,
            "total_operators": total_ops,
            "source_coverage": source_coverage,
            "target_coverage": target_coverage,
            "migration_gaps": list(migration_gaps),
            "compatibility_score": target_coverage,
        }
    
    def find_similar_operators(
        self,
        operator_id: str,
        threshold: float = 0.8
    ) -> List[Tuple[Entity, float]]:
        """Find operators similar to the given operator"""
        similar = []
        
        for neighbor, rel in self.get_neighbors(operator_id, RelationType.SIMILAR_TO, "both"):
            if rel.weight >= threshold:
                similar.append((neighbor, rel.weight))
        
        return sorted(similar, key=lambda x: x[1], reverse=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        return {
            "total_entities": len(self._entities),
            "total_relations": len(self._relations),
            "entities_by_type": {
                t.value: len(ids) for t, ids in self._entity_by_type.items()
            },
            "relations_by_type": {
                t.value: len(ids) for t, ids in self._relations_by_type.items()
            },
        }
    
    def to_dict(self) -> Dict:
        """Export knowledge graph to dictionary"""
        return {
            "entities": [e.to_dict() for e in self._entities.values()],
            "relations": [r.to_dict() for r in self._relations.values()],
            "statistics": self.get_statistics(),
        }
    
    def save(self, path: str):
        """Save knowledge graph to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved knowledge graph to {path}")
    
    @classmethod
    def load(cls, path: str) -> "MOHKG":
        """Load knowledge graph from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        kg = cls()
        
        for entity_data in data["entities"]:
            entity = Entity.from_dict(entity_data)
            kg.add_entity(entity)
        
        for relation_data in data["relations"]:
            kg.add_relation(
                source_id=relation_data["source_id"],
                target_id=relation_data["target_id"],
                relation_type=RelationType(relation_data["type"]),
                properties=relation_data.get("properties", {}),
                weight=relation_data.get("weight", 1.0),
            )
        
        logger.info(f"Loaded knowledge graph from {path}")
        return kg


class KGBuilder:
    """
    Builder for constructing MOH-KG from various data sources
    """
    
    def __init__(self):
        self.kg = MOHKG()
    
    def add_model(
        self,
        model_id: str,
        name: str,
        category: str,
        num_params: int,
        architecture: str,
        operators: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """Add a model with its operators to the knowledge graph"""
        # Add model entity
        model = ModelEntity(
            model_id=model_id,
            name=name,
            category=category,
            num_params=num_params,
            architecture=architecture,
            **kwargs,
        )
        self.kg.add_entity(model)
        
        # Add operators and relations
        for op_data in operators:
            op_id = op_data.get("id", f"{model_id}_{op_data['op_type']}_{len(self.kg._entities)}")
            
            operator = OperatorEntity(
                operator_id=op_id,
                name=op_data.get("name", op_data["op_type"]),
                op_type=op_data["op_type"],
                category=op_data.get("category", "other"),
            )
            self.kg.add_entity(operator)
            
            # Add CONTAINS relation
            self.kg.add_relation(
                source_id=model_id,
                target_id=op_id,
                relation_type=RelationType.CONTAINS,
            )
        
        return model_id
    
    def add_hardware(
        self,
        hardware_id: str,
        name: str,
        platform: str,
        vendor: str,
        compute_capability: str,
        memory_gb: float,
        supported_operators: List[str],
        **kwargs
    ) -> str:
        """Add a hardware platform with its supported operators"""
        # Add hardware entity
        hardware = HardwareEntity(
            hardware_id=hardware_id,
            name=name,
            platform=platform,
            vendor=vendor,
            compute_capability=compute_capability,
            memory_gb=memory_gb,
            **kwargs,
        )
        self.kg.add_entity(hardware)
        
        # Add SUPPORTED_BY relations for existing operators
        for op_type in supported_operators:
            # Find operators of this type
            for entity in self.kg.get_entities_by_type(EntityType.OPERATOR):
                if entity.properties.get("op_type") == op_type:
                    self.kg.add_relation(
                        source_id=entity.id,
                        target_id=hardware_id,
                        relation_type=RelationType.SUPPORTED_BY,
                    )
        
        return hardware_id
    
    def add_operator_similarity(
        self,
        operator1_id: str,
        operator2_id: str,
        similarity: float
    ):
        """Add similarity relation between operators"""
        self.kg.add_relation(
            source_id=operator1_id,
            target_id=operator2_id,
            relation_type=RelationType.SIMILAR_TO,
            weight=similarity,
        )
    
    def build(self) -> MOHKG:
        """Return the constructed knowledge graph"""
        stats = self.kg.get_statistics()
        logger.info(f"Built knowledge graph: {stats['total_entities']} entities, {stats['total_relations']} relations")
        return self.kg


class KGQueryEngine:
    """
    Query engine for MOH-KG with advanced query capabilities
    """
    
    def __init__(self, kg: MOHKG):
        self.kg = kg
    
    def query_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5
    ) -> List[List[str]]:
        """Find paths between two entities using BFS"""
        if start_id not in self.kg._entities or end_id not in self.kg._entities:
            return []
        
        from collections import deque
        
        queue = deque([(start_id, [start_id])])
        visited = {start_id}
        paths = []
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            if current == end_id:
                paths.append(path)
                continue
            
            for neighbor, rel in self.kg.get_neighbors(current, direction="both"):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append((neighbor.id, path + [neighbor.id]))
        
        return paths
    
    def query_subgraph(
        self,
        center_id: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """Extract subgraph around a center entity"""
        entities = {}
        relations = []
        
        def explore(entity_id: str, current_depth: int):
            if current_depth > depth or entity_id in entities:
                return
            
            entity = self.kg.get_entity(entity_id)
            if entity:
                entities[entity_id] = entity.to_dict()
            
            for neighbor, rel in self.kg.get_neighbors(entity_id, direction="both"):
                relations.append(rel.to_dict())
                explore(neighbor.id, current_depth + 1)
        
        explore(center_id, 0)
        
        return {
            "center": center_id,
            "depth": depth,
            "entities": list(entities.values()),
            "relations": relations,
        }
    
    def query_by_properties(
        self,
        entity_type: Optional[EntityType] = None,
        **property_filters
    ) -> List[Entity]:
        """Query entities by property values"""
        results = []
        
        if entity_type:
            candidates = self.kg.get_entities_by_type(entity_type)
        else:
            candidates = list(self.kg._entities.values())
        
        for entity in candidates:
            match = True
            for key, value in property_filters.items():
                if entity.properties.get(key) != value:
                    match = False
                    break
            if match:
                results.append(entity)
        
        return results


if __name__ == "__main__":
    # Test MOH-KG
    builder = KGBuilder()
    
    # Add sample model
    builder.add_model(
        model_id="llama-3.1-8b",
        name="Llama-3.1-8B",
        category="LLM",
        num_params=8_000_000_000,
        architecture="transformer",
        operators=[
            {"op_type": "MatMul", "category": "matrix"},
            {"op_type": "GELU", "category": "activation"},
            {"op_type": "LayerNorm", "category": "normalization"},
            {"op_type": "MultiHeadAttention", "category": "attention"},
            {"op_type": "RMSNorm", "category": "normalization"},
        ]
    )
    
    # Add hardware platforms
    builder.add_hardware(
        hardware_id="nvidia-a100",
        name="NVIDIA A100 80GB",
        platform="CUDA",
        vendor="NVIDIA",
        compute_capability="8.0",
        memory_gb=80,
        supported_operators=["MatMul", "GELU", "LayerNorm", "MultiHeadAttention", "RMSNorm"],
    )
    
    builder.add_hardware(
        hardware_id="ascend-910b",
        name="Huawei Ascend 910B",
        platform="CANN",
        vendor="Huawei",
        compute_capability="Ascend",
        memory_gb=64,
        supported_operators=["MatMul", "GELU", "LayerNorm", "MultiHeadAttention"],
    )
    
    # Build knowledge graph
    kg = builder.build()
    
    # Print statistics
    stats = kg.get_statistics()
    print(f"Knowledge Graph Statistics:")
    print(f"  Entities: {stats['total_entities']}")
    print(f"  Relations: {stats['total_relations']}")
    print(f"  By type: {stats['entities_by_type']}")
    
    # Test migration compatibility
    compatibility = kg.calculate_migration_compatibility(
        model_id="llama-3.1-8b",
        source_hardware_id="nvidia-a100",
        target_hardware_id="ascend-910b",
    )
    print(f"\nMigration Compatibility:")
    print(f"  Source coverage: {compatibility['source_coverage']:.2%}")
    print(f"  Target coverage: {compatibility['target_coverage']:.2%}")
    print(f"  Migration gaps: {compatibility['migration_gaps']}")
