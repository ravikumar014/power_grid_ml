"""
Graph-based feature engineering for the power-grid project.

The electrical grid is represented as a graph:

    G = (V, E)

where:

    V = buses / electrical nodes
    E = transmission or distribution lines

This module derives topology-aware features from the grid structure.

Node features
-------------
- Degree
- Weighted degree
- Betweenness centrality
- Closeness centrality
- Eigenvector centrality
- Local clustering coefficient
- PageRank

Edge features
-------------
- Edge flow magnitude
- Flow direction
- Flow imbalance
- Edge loading
- Edge loading ratio
- Endpoint degree
- Endpoint centrality

Important
---------
This module does NOT implement a Graph Neural Network.

It creates topology-aware numerical features that can be appended to
the conventional statistical/electrical/frequency features.

Later, Milestone 6 will use the same graph representation for:

    GCN
    GraphSAGE
    GAT
    Graph Transformer
    Graph Autoencoder
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp


                                                                        
               
                                                                        


@dataclass
class GraphFeatureConfig:
    """
    Configuration for graph feature extraction.
    """

    directed: bool = False

    include_degree: bool = True

    include_weighted_degree: bool = True

    include_betweenness: bool = True

    include_closeness: bool = True

    include_eigenvector: bool = True

    include_clustering: bool = True

    include_pagerank: bool = True

    include_edge_features: bool = True

    epsilon: float = 1e-8


                                                                        
                        
                                                                        


class GraphFeatureEngineer:
    """
    Build a power-grid graph and derive topology-aware features.
    """

    def __init__(
        self,
        config: GraphFeatureConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else GraphFeatureConfig()
        )

        self.graph: nx.Graph | nx.DiGraph | None = None

                                                                        
                
                                                                        

    @staticmethod
    def _validate_columns(
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> None:

        missing = (
            set(columns)
            - set(dataframe.columns)
        )

        if missing:

            raise ValueError(
                "Required columns missing: "
                f"{sorted(missing)}"
            )

                                                                        
                           
                                                                        

    @staticmethod
    def load_network_topology(
        path: str | Path,
        active_only: bool = True,
    ) -> pd.DataFrame:
        """
        Load the actual grid topology from a pandapower network JSON.

        The project already uses pandapower to construct the network,
        so we use pandapower itself to deserialize the network rather
        than manually parsing pandapower's internal JSON structure.

        Parameters
        ----------
        path:
            Path to the pandapower network JSON.

        active_only:
            If True, retain only lines where in_service=True.

        Returns
        -------
        DataFrame
            line_id
            from_bus
            to_bus
            in_service
        """

        path = Path(path)

        if not path.exists():

            raise FileNotFoundError(
                f"Network file not found: {path}"
            )

                                                                        
                                        
                                                                        

        net = pp.from_json(
            str(path)
        )

                                                                        
                                          
                                                                        

        if not hasattr(net, "line"):

            raise ValueError(
                "Loaded pandapower network does not "
                "contain a line table."
            )

        if net.line.empty:

            raise ValueError(
                "Loaded pandapower network contains "
                "no transmission/distribution lines."
            )

                                                                        
                                            
                                                                        

        required = [
            "from_bus",
            "to_bus",
            "in_service",
        ]

        missing = (
            set(required)
            - set(net.line.columns)
        )

        if missing:

            raise ValueError(
                "Network line table is missing "
                f"columns: {sorted(missing)}"
            )

        topology = net.line[
            required
        ].copy()

                                                                        
                                                           
                                                                        

        topology.index.name = "line_id"

        topology = topology.reset_index()

                                                                        
                                
                                                                        

        topology[
            "line_id"
        ] = topology[
            "line_id"
        ].astype(int)

        topology[
            "from_bus"
        ] = topology[
            "from_bus"
        ].astype(int)

        topology[
            "to_bus"
        ] = topology[
            "to_bus"
        ].astype(int)

        topology[
            "in_service"
        ] = topology[
            "in_service"
        ].astype(bool)

                                                                        
                                      
         
                                                             
                           
                                                                        

        if active_only:

            topology = topology[
                topology[
                    "in_service"
                ]
            ].copy()

        topology = topology.reset_index(
            drop=True
        )

        return topology

                                                                        
                 
                                                                        

    def build_graph(
        self,
        topology_dataframe: pd.DataFrame,
        from_column: str = "from_bus",
        to_column: str = "to_bus",
        line_id_column: str = "line_id",
        bus_ids: list[int] | None = None,
    ) -> nx.Graph | nx.DiGraph:
        """
        Construct the static grid topology from a topology dataframe.

        The topology dataframe is intentionally separate from the
        time-series measurement dataframe.

        Required columns
        ----------------
        line_id
        from_bus
        to_bus
        """

        required = [
            line_id_column,
            from_column,
            to_column,
        ]

        self._validate_columns(
            topology_dataframe,
            required,
        )

        if self.config.directed:

            graph = nx.DiGraph()

        else:

            graph = nx.Graph()
            if bus_ids is not None:

                graph.add_nodes_from(
                    [
                        self._normalize_node_id(
                            bus_id
                        )
                        for bus_id in bus_ids
                    ]
                )

        for _, row in topology_dataframe.iterrows():

            source = row[
                from_column
            ]

            target = row[
                to_column
            ]

            line_id = row[
                line_id_column
            ]

            if (
                pd.isna(source)
                or pd.isna(target)
            ):
                continue

            source = self._normalize_node_id(
                source
            )

            target = self._normalize_node_id(
                target
            )

            line_id = self._normalize_node_id(
                line_id
            )

            if source == target:
                continue

            graph.add_edge(
                source,
                target,
                line_id=line_id,
            )

        self.graph = graph

        return graph
    

                   
                        
                           
                        
             
                                             
             

                           

                               

                                      
                                                    
               

                                 
                  
           

                      
                        
                         
                       
           

                     
                           
                                     
           

                     

                               
                                                      
                                      
               

                         

                                                                        
                           
                                                                        

    @staticmethod
    def _normalize_node_id(
        value,
    ):
        """
        Normalize node identifiers while preserving their identity.

        Integer-like floating point IDs such as 3.0 become 3.
        """

        if isinstance(
            value,
            (float, np.floating),
        ):

            if np.isfinite(value) and value.is_integer():

                return int(value)

        return value

                                                                        
                            
                                                                        

    def calculate_node_features(
        self,
        graph: nx.Graph | nx.DiGraph | None = None,
    ) -> pd.DataFrame:
        """
        Calculate topology features for every node.
        """

        if graph is None:

            graph = self.graph

        if graph is None:

            raise RuntimeError(
                "Graph has not been built. "
                "Call build_graph() first."
            )

        nodes = list(
            graph.nodes()
        )

        result = pd.DataFrame(
            index=nodes
        )

                                                                        
                
                                                                        

        if self.config.include_degree:

            degree = dict(
                graph.degree()
            )

            result[
                "node_degree"
            ] = pd.Series(
                degree
            )

                                                                        
                         
         
                                                               
                                                                        

        if self.config.include_weighted_degree:

            weighted_degree = dict(
                graph.degree(
                    weight="weight"
                )
            )

            result[
                "node_weighted_degree"
            ] = pd.Series(
                weighted_degree
            )

                                                                        
                                
                                                                        

        if self.config.include_betweenness:

            betweenness = nx.betweenness_centrality(
                graph,
                normalized=True,
            )

            result[
                "betweenness_centrality"
            ] = pd.Series(
                betweenness
            )

                                                                        
                              
                                                                        

        if self.config.include_closeness:

            closeness = nx.closeness_centrality(
                graph
            )

            result[
                "closeness_centrality"
            ] = pd.Series(
                closeness
            )

                                                                        
                                
                                                                        

        if self.config.include_eigenvector:

            try:

                eigenvector = (
                    nx.eigenvector_centrality(
                        graph,
                        max_iter=1000,
                    )
                )

            except nx.PowerIterationFailedConvergence:

                eigenvector = {
                    node: np.nan
                    for node in nodes
                }

            result[
                "eigenvector_centrality"
            ] = pd.Series(
                eigenvector
            )

                                                                        
                                      
                                                                        

        if self.config.include_clustering:

                                                                     
                                                                    
            clustering_graph = (
                graph
                if not graph.is_directed()
                else graph.to_undirected()
            )

            clustering = (
                nx.clustering(
                    clustering_graph
                )
            )

            result[
                "local_clustering_coefficient"
            ] = pd.Series(
                clustering
            )

                                                                        
                  
                                                                        

        if self.config.include_pagerank:

            try:

                pagerank = nx.pagerank(
                    graph
                )

            except nx.NetworkXException:

                pagerank = {
                    node: np.nan
                    for node in nodes
                }

            result[
                "pagerank"
            ] = pd.Series(
                pagerank
            )

        result.index.name = "bus_id"

        return result.reset_index()

                                                                        
                            
                                                                        

    def calculate_edge_topology_features(
        self,
        graph: nx.Graph | nx.DiGraph | None = None,
    ) -> pd.DataFrame:
        """
        Calculate structural features for every grid edge.
        """

        if graph is None:

            graph = self.graph

        if graph is None:

            raise RuntimeError(
                "Graph has not been built."
            )

        degree = dict(
            graph.degree()
        )

        betweenness = (
            nx.edge_betweenness_centrality(
                graph,
                normalized=True,
            )
        )

        rows = []

        for source, target in graph.edges():

            edge_key = (
                source,
                target,
            )

            row = {
                "from_bus": source,
                "to_bus": target,
                "from_bus_degree": (
                    degree.get(
                        source,
                        np.nan,
                    )
                ),
                "to_bus_degree": (
                    degree.get(
                        target,
                        np.nan,
                    )
                ),
                "edge_betweenness_centrality": (
                    betweenness.get(
                        edge_key,
                        np.nan,
                    )
                ),
            }

            rows.append(
                row
            )

        return pd.DataFrame(
            rows
        )

                                                                        
                   
                                                                        

    def calculate_flow_features(
        self,
        line_dataframe: pd.DataFrame,
        from_bus_column: str = "from_bus",
        to_bus_column: str = "to_bus",
        current_from_column: str = "current_from_ka",
        current_to_column: str = "current_to_ka",
        loading_column: str = "loading_percent",
    ) -> pd.DataFrame:
        """
        Calculate dynamic flow features for each line measurement.

        The input dataframe must already contain topology columns:

            from_bus
            to_bus

        together with the dynamic measurements:

            current_from_ka
            current_to_ka
            loading_percent

        Topology is joined to the measurement dataframe in
        transform_line() using line_id.
        """

        required = [
            from_bus_column,
            to_bus_column,
            current_from_column,
            current_to_column,
            loading_column,
        ]

        self._validate_columns(
            line_dataframe,
            required,
        )

        data = line_dataframe.copy()

                                                                        
                            
                                                                        

        for column in [
            current_from_column,
            current_to_column,
            loading_column,
        ]:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

        current_from = (
            data[
                current_from_column
            ].abs()
        )

        current_to = (
            data[
                current_to_column
            ].abs()
        )

        loading = data[
            loading_column
        ]

                                                                        
                                
         
                                          
                                                                        

        mean_current = (
            current_from
            + current_to
        ) / 2.0

                                                                        
                         
         
                                
                                                                        

        flow_difference = (
            current_from
            - current_to
        )

        absolute_flow_difference = (
            flow_difference.abs()
        )

                                                                        
                        
         
                     
                         
                          
                               
                                                                        

        denominator = (
            pd.concat(
                [
                    current_from,
                    current_to,
                ],
                axis=1,
            )
            .max(axis=1)
            + self.config.epsilon
        )

        flow_imbalance = (
            absolute_flow_difference
            / denominator
        )

                                                                        
                       
         
                               
                                                                        

        loading_ratio = (
            loading
            / 100.0
        )

                                                                        
                        
                                                                        

        loading_stress = (
            loading_ratio
        )

                                                                        
                           
         
                                                                  
                                           
                                                                        

        result = pd.DataFrame(
            {
                "mean_edge_current_ka": (
                    mean_current.values
                ),

                "edge_flow_difference_ka": (
                    flow_difference.values
                ),

                "absolute_edge_flow_difference_ka": (
                    absolute_flow_difference.values
                ),

                "edge_flow_imbalance": (
                    flow_imbalance.values
                ),

                "edge_loading_ratio": (
                    loading_ratio.values
                ),

                "edge_loading_stress": (
                    loading_stress.values
                ),

                "edge_overloaded": (
                    (loading > 100.0)
                    .astype(int)
                    .values
                ),
            },
            index=line_dataframe.index,
        )

        return result

                                                                        
                                              
                                                                        

    def transform_bus(
        self,
        bus_dataframe: pd.DataFrame,
        bus_id_column: str = "bus_id",
    ) -> pd.DataFrame:
        """
        Attach static topology features to every bus measurement.

        The topology features are constant for a static grid, while
        the electrical measurements vary with time.
        """

        if bus_id_column not in bus_dataframe.columns:

            raise ValueError(
                f"Missing bus identifier column: "
                f"{bus_id_column}"
            )

        node_features = (
            self.calculate_node_features()
        )

        result = bus_dataframe.copy()

                                                                        
                                    
                                                                        

        result["_graph_bus_id"] = (
            result[bus_id_column]
            .apply(
                self._normalize_node_id
            )
        )

        node_features["_graph_bus_id"] = (
            node_features["bus_id"]
            .apply(
                self._normalize_node_id
            )
        )

        result = result.merge(
            node_features.drop(
                columns="bus_id"
            ),
            how="left",
            on="_graph_bus_id",
        )

        result = result.drop(
            columns="_graph_bus_id"
        )

        return result

                                                                        
                                               
                                                                        

    def transform_line(
        self,
        line_dataframe: pd.DataFrame,
        topology_dataframe: pd.DataFrame,
        line_id_column: str = "line_id",
        from_bus_column: str = "from_bus",
        to_bus_column: str = "to_bus",
    ) -> pd.DataFrame:
        """
        Attach static topology features and dynamic flow features
        to every line measurement.

        The measurement dataframe contains:

            timestamp
            line_id
            current_from_ka
            current_to_ka
            loading_percent

        The topology dataframe contains:

            line_id
            from_bus
            to_bus
            in_service

        They are joined using line_id.
        """

                                                                        
                                         
                                                                        

        self._validate_columns(
            line_dataframe,
            [
                line_id_column,
                "current_from_ka",
                "current_to_ka",
                "loading_percent",
            ],
        )

                                                                        
                                      
                                                                        

        self._validate_columns(
            topology_dataframe,
            [
                line_id_column,
                from_bus_column,
                to_bus_column,
            ],
        )

                                                                        
                                   
                                                                        

        topology_features = (
            self.calculate_edge_topology_features()
        )

                                                                        
                                   
         
                                    
                                                                        

        topology_mapping = (
            topology_dataframe[
                [
                    line_id_column,
                    from_bus_column,
                    to_bus_column,
                    "in_service",
                ]
            ]
            .drop_duplicates(
                subset=[line_id_column]
            )
            .copy()
        )

                                                                        
                             
                                                                        

        topology_mapping[
            line_id_column
        ] = pd.to_numeric(
            topology_mapping[
                line_id_column
            ],
            errors="coerce",
        )

                                                                        
                                                   
                                                                        

        data = line_dataframe.merge(
            topology_mapping,
            how="left",
            on=line_id_column,
            validate="many_to_one",
        )

                                                                        
                                                           
                                                                        

        data["in_service"] = (
            data["in_service"]
            .astype(bool)
        )

                                                                        
                                                               
                                                                        

        missing_topology = data[
            from_bus_column
        ].isna() | data[
            to_bus_column
        ].isna()

        if missing_topology.any():

            missing_ids = (
                data.loc[
                    missing_topology,
                    line_id_column,
                ]
                .dropna()
                .unique()
                .tolist()
            )

            raise ValueError(
                "Some line measurements could not be "
                "mapped to grid topology. "
                f"Missing line IDs: {missing_ids}"
            )

                                                                        
                                
                                                                        

        flow_features = (
            self.calculate_flow_features(
                data,
                from_bus_column=from_bus_column,
                to_bus_column=to_bus_column,
            )
        )

                                                                        
                                           
                                                                        

        topology_features[
            "_topology_from_bus"
        ] = (
            topology_features[
                "from_bus"
            ].apply(
                self._normalize_node_id
            )
        )

        topology_features[
            "_topology_to_bus"
        ] = (
            topology_features[
                "to_bus"
            ].apply(
                self._normalize_node_id
            )
        )

                                                                        
                                        
                                                                        

        data[
            "_graph_from_bus"
        ] = (
            data[
                from_bus_column
            ].apply(
                self._normalize_node_id
            )
        )

        data[
            "_graph_to_bus"
        ] = (
            data[
                to_bus_column
            ].apply(
                self._normalize_node_id
            )
        )

                                                                        
                                              
                                                                        

        topology_features = (
            topology_features.drop(
                columns=[
                    "from_bus",
                    "to_bus",
                ]
            )
        )

        result = data.merge(
            topology_features,
            how="left",
            left_on=[
                "_graph_from_bus",
                "_graph_to_bus",
            ],
            right_on=[
                "_topology_from_bus",
                "_topology_to_bus",
            ],
        )

                                                                        
                                                                      
                                                                        

        result["edge_in_operational_graph"] = (
            result["in_service"]
            .astype(int)
        )

                                                                        
                                         
                                                                        

        result = result.drop(
            columns=[
                "_graph_from_bus",
                "_graph_to_bus",
                "_topology_from_bus",
                "_topology_to_bus",
            ],
            errors="ignore",
        )

                                                                        
                                       
                                                                        

        result = pd.concat(
            [
                result.reset_index(
                    drop=True
                ),
                flow_features.reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )

        return result

                                                                        
          
                                                                        

    @staticmethod
    def save(
        dataframe: pd.DataFrame,
        path: str | Path,
    ) -> Path:
        """
        Save graph features.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_csv(
            path,
            index=False,
        )

        return path


                                                                        
      
                                                                        


if __name__ == "__main__":

    from src.eda.load_data import (
        EDADataLoader,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "GRAPH FEATURE ENGINEERING"
    )

    print(
        "=" * 70
    )

                                                                    
               
                                                                    

    loader = EDADataLoader()

    dataset = loader.load_all()

                                                                    
                             
                                                                    

    print(
        "\nBus columns:"
    )

    print(
        list(dataset.bus.columns)
    )

    print(
        "\nLine columns:"
    )

    print(
        list(dataset.line.columns)
    )

                                                                    
                   
                                                                    

    config = GraphFeatureConfig(
        directed=False,
        include_degree=True,
        include_weighted_degree=True,
        include_betweenness=True,
        include_closeness=True,
        include_eigenvector=True,
        include_clustering=True,
        include_pagerank=True,
        include_edge_features=True,
    )

    engineer = (
        GraphFeatureEngineer(
            config=config
        )
    )

                                                                    
                    
     
                
     
                                           
     
                  
                
     
                                                                      
            
                                                                    

                                                                    
                                              
                                                                    

    network_path = Path(
        "data/simulated/networks/ieee33.json"
    )

    topology = (
        engineer.load_network_topology(
            network_path,
            active_only=False,
        )
    )

    print(
        "\nLoaded network topology:"
    )

    print(
        topology.to_string(
            index=False
        )
    )

    print(
        f"\nActive lines: "
        f"{len(topology)}"
    )

    print(
        f"Active lines: "
        f"{topology['in_service'].sum()}"
    )

    print(
        f"Inactive / normally-open lines: "
        f"{(~topology['in_service']).sum()}"
    )

                                                                    
                          
     
                                                                 
                                                                    

    active_topology = (
        topology[
            topology["in_service"]
        ].copy()
    )

    graph = engineer.build_graph(
        active_topology,
        line_id_column="line_id",
        from_column="from_bus",
        to_column="to_bus",
        bus_ids=(
            dataset.bus[
                "bus_id"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
    )

    print()
    print(
        "Operational graph summary:"
    )

    print(
        f"Nodes: "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"Edges: "
        f"{graph.number_of_edges()}"
    )

    print(
        f"Connected components: "
        f"{nx.number_connected_components(graph)}"
    )

                                                                    
                 
                                                                    

    graph = engineer.build_graph(
        topology,
        line_id_column="line_id",
        from_column="from_bus",
        to_column="to_bus",
        bus_ids=(
            dataset.bus[
                "bus_id"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
    )

    print()
    print(
        "Graph summary:"
    )

    print(
        f"Nodes: "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"Edges: "
        f"{graph.number_of_edges()}"
    )

    print(
        f"Connected components: "
        f"{nx.number_connected_components(graph)}"
    )

    print(
        "\nTopology:"
    )

    print(
        topology.head(
            10
        ).to_string(
            index=False
        )
    )

                                                                    
                   
                                                                    

    node_features = (
        engineer.calculate_node_features(
            graph
        )
    )

    node_path = engineer.save(
        node_features,
        "data/features/graph/"
        "node_features.csv",
    )

    print(
        f"\nSaved node features to:"
        f"\n{node_path}"
    )

    print(
        "\nNode features:"
    )

    print(
        node_features.head(
            10
        ).to_string(
            index=False
        )
    )

                                                                    
                   
                                                                    

    edge_topology = (
        engineer.calculate_edge_topology_features(
            graph
        )
    )

    edge_topology_path = engineer.save(
        edge_topology,
        "data/features/graph/"
        "edge_topology_features.csv",
    )

    print(
        f"\nSaved edge topology features to:"
        f"\n{edge_topology_path}"
    )

                                                                    
                        
                                                                    

    bus_graph_features = (
        engineer.transform_bus(
            dataset.bus
        )
    )

    bus_path = engineer.save(
        bus_graph_features,
        "data/features/bus/"
        "graph_features.csv",
    )

    print(
        f"\nSaved bus graph features to:"
        f"\n{bus_path}"
    )

    print(
        f"\nBus graph feature matrix shape:"
        f" {bus_graph_features.shape}"
    )

                                                                    
                         
                                                                    

    line_graph_features = (
        engineer.transform_line(
            dataset.line,
            topology,
        )
    )

    line_path = engineer.save(
        line_graph_features,
        "data/features/line/"
        "graph_features.csv",
    )

    print(
        f"\nSaved line graph features to:"
        f"\n{line_path}"
    )

    print(
        f"\nLine graph feature matrix shape:"
        f" {line_graph_features.shape}"
    )

    print()
    print(
        "=" * 70
    )

    print(
        "GRAPH FEATURE ENGINEERING COMPLETE"
    )

    print(
        "=" * 70
    )