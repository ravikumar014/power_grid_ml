"""
Power-grid network generation and validation.

This module creates the base electrical network used by the
simulation pipeline.

Current target:
    IEEE 33-bus distribution network

Responsibilities:
    1. Create the network.
    2. Validate basic topology.
    3. Provide network summary information.
    4. Save/load the network representation.

Measurement generation and anomaly injection are handled elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import pandapower as pp
import pandapower.networks as pn


class GridGenerator:
    """
    Generate and validate benchmark power-grid networks.
    """

    SUPPORTED_NETWORKS = {
        "ieee33": pn.case33bw,
    }

    def __init__(self, network_name: str = "ieee33") -> None:
        """
        Parameters
        ----------
        network_name:
            Name of the benchmark network to generate.
        """

        network_name = network_name.lower()

        if network_name not in self.SUPPORTED_NETWORKS:
            supported = ", ".join(self.SUPPORTED_NETWORKS.keys())

            raise ValueError(
                f"Unsupported network '{network_name}'. "
                f"Supported networks: {supported}"
            )

        self.network_name = network_name
        self.net: pp.pandapowerNet | None = None

    def generate(self) -> pp.pandapowerNet:
        """
        Generate the selected benchmark network.

        Returns
        -------
        pandapowerNet
            The generated pandapower network.
        """

        generator = self.SUPPORTED_NETWORKS[self.network_name]

        self.net = generator()

        return self.net

    def validate(self) -> bool:
        """
        Perform basic structural validation of the generated network.

        Returns
        -------
        bool
            True if the network passes validation.

        Raises
        ------
        RuntimeError
            If the network has not been generated yet.
        ValueError
            If basic topology requirements are violated.
        """

        if self.net is None:
            raise RuntimeError(
                "Network has not been generated. "
                "Call generate() before validate()."
            )

        net = self.net

                                                                   
                                
                                                                   

        if len(net.bus) == 0:
            raise ValueError("Network contains no buses.")

        if len(net.line) == 0:
            raise ValueError("Network contains no transmission/distribution lines.")

        if len(net.ext_grid) == 0 and len(net.gen) == 0:
            raise ValueError(
                "Network contains neither an external grid nor a generator."
            )

                                                                   
                                            
                                                                   

        bus_ids = set(net.bus.index)

        for line_id, line in net.line.iterrows():

            from_bus = line["from_bus"]
            to_bus = line["to_bus"]

            if from_bus not in bus_ids:
                raise ValueError(
                    f"Line {line_id} references invalid from_bus={from_bus}."
                )

            if to_bus not in bus_ids:
                raise ValueError(
                    f"Line {line_id} references invalid to_bus={to_bus}."
                )

                                                                   
                              
                                                                   

        graph = self.to_networkx()

                                               
        if graph.number_of_nodes() != len(net.bus):
            raise ValueError(
                "Topology graph does not contain all network buses."
            )

                                                                   
                            
                                                                   

        if not nx.is_connected(graph):
            raise ValueError(
                "Network topology is disconnected."
            )

        return True

    def to_networkx(self) -> nx.Graph:
        """
        Convert the pandapower network topology into a NetworkX graph.

        Nodes:
            Electrical buses.

        Edges:
            Active electrical lines.

        Returns
        -------
        networkx.Graph
            Graph representation of the grid.
        """

        if self.net is None:
            raise RuntimeError(
                "Network has not been generated. "
                "Call generate() first."
            )

        graph = nx.Graph()

                        
        for bus_id in self.net.bus.index:
            graph.add_node(
                int(bus_id),
                name=str(self.net.bus.loc[bus_id, "name"]),
                voltage_kv=float(self.net.bus.loc[bus_id, "vn_kv"]),
            )

                         
        for line_id, line in self.net.line.iterrows():

            from_bus = int(line["from_bus"])
            to_bus = int(line["to_bus"])

            graph.add_edge(
                from_bus,
                to_bus,
                line_id=int(line_id),
                length_km=float(line["length_km"]),
                r_ohm_per_km=float(line["r_ohm_per_km"]),
                x_ohm_per_km=float(line["x_ohm_per_km"]),
                max_i_ka=float(line["max_i_ka"]),
            )

        return graph

    def summary(self) -> dict[str, Any]:
        """
        Return a compact summary of the generated network.
        """

        if self.net is None:
            raise RuntimeError(
                "Network has not been generated. "
                "Call generate() first."
            )

        graph = self.to_networkx()

        return {
            "network": self.network_name,
            "num_buses": len(self.net.bus),
            "num_lines": len(self.net.line),
            "num_loads": len(self.net.load),
            "num_generators": len(self.net.gen),
            "num_external_grids": len(self.net.ext_grid),
            "num_static_generators": len(self.net.sgen),
            "num_switches": len(self.net.switch),
            "connected": nx.is_connected(graph),
            "num_graph_nodes": graph.number_of_nodes(),
            "num_graph_edges": graph.number_of_edges(),
        }

    def save(self, path: str | Path) -> None:
        """
        Save the generated pandapower network as JSON.

        Parameters
        ----------
        path:
            Destination JSON file.
        """

        if self.net is None:
            raise RuntimeError(
                "Network has not been generated. "
                "Call generate() first."
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        pp.to_json(self.net, str(path))

    @staticmethod
    def load(path: str | Path) -> pp.pandapowerNet:
        """
        Load a previously saved pandapower network.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Network file not found: {path}"
            )

        return pp.from_json(str(path))


if __name__ == "__main__":

    generator = GridGenerator(network_name="ieee33")

    network = generator.generate()

    generator.validate()

    print("\nNetwork generated successfully.\n")

    print("Network summary:")
    print("-" * 40)

    for key, value in generator.summary().items():
        print(f"{key:25s}: {value}")

    output_path = Path("data/simulated/networks/ieee33.json")

    generator.save(output_path)

    print(f"\nNetwork saved to: {output_path}")