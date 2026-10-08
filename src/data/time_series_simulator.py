"""
Power-system time-series simulation.

This module connects:
    - the static pandapower grid
    - time-varying load profiles
    - time-varying DER profiles

and performs a power-flow calculation for every timestep.

Outputs:
    - bus voltage magnitude
    - bus voltage angle
    - bus active/reactive power
    - line current
    - line loading
    - load/generation measurements

The output of this module forms the first genuine electrical
measurement dataset used by the ML pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp

                                             
                                                  
                                                                  

from src.data.grid_generator import GridGenerator


class TimeSeriesSimulator:
    """
    Run time-series power-flow simulations.

    Parameters
    ----------
    network_name:
        Benchmark network name.

    profile_path:
        Path to the generated load/solar profile CSV.

    der_bus:
        Bus where the simulated solar DER is connected.

    solar_capacity_mw:
        Installed solar capacity.

    """

    def __init__(
        self,
        network_name: str = "ieee33",
        profile_path: str | Path = (
            "data/simulated/profiles/profiles.csv"
        ),
        der_bus: int = 18,
        solar_capacity_mw: float = 0.5,
    ) -> None:

        self.network_name = network_name
        self.profile_path = Path(profile_path)

        self.der_bus = der_bus
        self.solar_capacity_mw = solar_capacity_mw

        self.net: pp.pandapowerNet | None = None
        self.profiles: pd.DataFrame | None = None

                                                                        
                  
                                                                        

    def load_network(self) -> pp.pandapowerNet:
        """
        Generate the benchmark network.
        """

        generator = GridGenerator(
            network_name=self.network_name
        )

        self.net = generator.generate()

        generator.validate()

        return self.net

                                                                        
                   
                                                                        

    def load_profiles(self) -> pd.DataFrame:
        """
        Load the previously generated temporal profiles.
        """

        if not self.profile_path.exists():
            raise FileNotFoundError(
                f"Profile file not found: {self.profile_path}"
            )

        profiles = pd.read_csv(
            self.profile_path,
            parse_dates=["timestamp"],
        )

        required_columns = {
            "timestamp",
            "load_pu",
            "solar_pu",
        }

        missing = required_columns.difference(
            profiles.columns
        )

        if missing:
            raise ValueError(
                "Profile dataset is missing required columns: "
                f"{sorted(missing)}"
            )

        if profiles.empty:
            raise ValueError(
                "Profile dataset contains no rows."
            )

        if profiles["timestamp"].duplicated().any():
            raise ValueError(
                "Profile dataset contains duplicate timestamps."
            )

        if not profiles["timestamp"].is_monotonic_increasing:
            raise ValueError(
                "Profile timestamps are not monotonically increasing."
            )

        self.profiles = profiles

        return profiles

                                                                        
             
                                                                        

    def add_solar_der(self) -> int:
        """
        Add a solar PV static generator to the network.

        Returns
        -------
        int
            Index of the newly created solar generator.
        """

        if self.net is None:
            raise RuntimeError(
                "Network has not been loaded."
            )

        if self.der_bus not in self.net.bus.index:
            raise ValueError(
                f"DER bus {self.der_bus} does not exist "
                f"in the network."
            )

        der_index = pp.create_sgen(
            self.net,
            bus=self.der_bus,
            p_mw=0.0,
            q_mvar=0.0,
            name="Solar_PV_1",
        )

        return int(der_index)

                                                                        
                             
                                                                        

    def create_load_controller(self) -> None:
        """
        Create a pandapower controller that scales every load
        according to the common temporal load profile.
        """

        if self.net is None:
            raise RuntimeError(
                "Network has not been loaded."
            )

        if self.profiles is None:
            raise RuntimeError(
                "Profiles have not been loaded."
            )

        load_indices = list(
            self.net.load.index
        )

        if not load_indices:
            raise ValueError(
                "The network contains no loads."
            )

                                                                        
                                                
         
                                              
         
                                                   
                                                                        

        p_profiles = pd.DataFrame(
            index=range(len(self.profiles))
        )

        for load_idx in load_indices:

            base_p = float(
                self.net.load.at[
                    load_idx,
                    "p_mw",
                ]
            )

            p_profiles[load_idx] = (
                base_p
                * self.profiles["load_pu"].to_numpy()
            )

        p_profiles = p_profiles[
            load_indices
        ]

        p_source = DFData(
            p_profiles
        )

        ConstControl(
            self.net,
            element="load",
            element_index=load_indices,
            variable="p_mw",
            data_source=p_source,
            profile_name=load_indices,
        )

                                                                        
                        
         
                                                               
                                                                        

        q_profiles = pd.DataFrame(
            index=range(len(self.profiles))
        )

        for load_idx in load_indices:

            base_q = float(
                self.net.load.at[
                    load_idx,
                    "q_mvar",
                ]
            )

            q_profiles[load_idx] = (
                base_q
                * self.profiles["load_pu"].to_numpy()
            )

        q_profiles = q_profiles[
            load_indices
        ]

        q_source = DFData(
            q_profiles
        )

        ConstControl(
            self.net,
            element="load",
            element_index=load_indices,
            variable="q_mvar",
            data_source=q_source,
            profile_name=load_indices,
        )

                                                                        
                           
                                                                        

    def create_der_controller(
        self,
        der_index: int,
    ) -> None:
        """
        Create a controller for the solar PV generator.
        """

        if self.net is None:
            raise RuntimeError(
                "Network has not been loaded."
            )

        if self.profiles is None:
            raise RuntimeError(
                "Profiles have not been loaded."
            )

        solar_profile = (
            self.solar_capacity_mw
            * self.profiles["solar_pu"].to_numpy()
        )

        der_profiles = pd.DataFrame(
            {
                der_index: solar_profile
            }
        )

        der_source = DFData(
            der_profiles
        )

        ConstControl(
            self.net,
            element="sgen",
            element_index=[der_index],
            variable="p_mw",
            data_source=der_source,
            profile_name=[der_index],
        )

                                                                        
                     
                                                                        

    def collect_results(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert pandapower result tables into ML-friendly
        long-format dataframes.

        Returns
        -------
        bus_measurements:
            One row per timestep and bus.

        line_measurements:
            One row per timestep and line.
        """

        if self.net is None:
            raise RuntimeError(
                "Network has not been loaded."
            )

        if self.profiles is None:
            raise RuntimeError(
                "Profiles have not been loaded."
            )

        bus_records: list[dict] = []
        line_records: list[dict] = []

        timestamps = self.profiles[
            "timestamp"
        ].tolist()

                                                                        
                                                    
         
                                                               
                                                                   
                                                                        

        if not hasattr(
            self,
            "_bus_results",
        ):
            raise RuntimeError(
                "No simulation results available."
            )

                                                                        
                     
                                                                        

        for timestep, result in enumerate(
            self._bus_results
        ):

            timestamp = timestamps[timestep]

            for bus_id, row in result.iterrows():

                bus_records.append(
                    {
                        "timestamp": timestamp,
                        "bus_id": int(bus_id),
                        "voltage_pu": float(
                            row["vm_pu"]
                        ),
                        "angle_deg": float(
                            row["va_degree"]
                        ),
                        "active_power_mw": float(
                            row["p_mw"]
                        ),
                        "reactive_power_mvar": float(
                            row["q_mvar"]
                        ),
                    }
                )

                                                                        
                      
                                                                        

        for timestep, result in enumerate(
            self._line_results
        ):

            timestamp = timestamps[timestep]

            for line_id, row in result.iterrows():

                line_records.append(
                    {
                        "timestamp": timestamp,
                        "line_id": int(line_id),
                        "current_from_ka": float(
                            row["i_from_ka"]
                        ),
                        "current_to_ka": float(
                            row["i_to_ka"]
                        ),
                        "loading_percent": float(
                            row["loading_percent"]
                        ),
                    }
                )

        return (
            pd.DataFrame(bus_records),
            pd.DataFrame(line_records),
        )

                                                                        
                    
                                                                        

              
               
                                             
             
                                                  
             

                             
                              

                                     
                                          

                                                                          
                         
                                                                          

                                          

                                                                          
                              
                                                                          

                                       

                                     
                                 
           

                                                                          
                                                         
           
                                                                     
                                                                     
                                                                          

                                
                                 

                                
                                
            

                                                    
                                                                

                                                       
                                      
                   

                                                              
                                              
                      

                                                                          
                                                                   
                                                                   
                        
                                                                          

                                 
                                     
                              
                              
               

                                  
                                     
                              
                               
               

                             
                                                  

                                   
                               
                             
                       
                                       
                                   
                                          
                       
                                        
                                              
                                            
                                   
                                 
                       
                   

                                                                          
                                               
                                                                          

                  
                                 
                                              
                

                                
                                     
                                    
                            
                          

                                
                                       
                                    
                              
                          

                                   
                            
                        
                   
                                
                                     
                   
                              
               

                            
                          
                   
                                
                                       
                   
                              
               

                                                                          
                                    
                                                                          

                               
                            
                         
                   
                                        
                               
               

                                                                          
                                  
                                                                          

                  

                           
                               
                                     
                   

                                      

                                                 
                              
                                

                                     
                                                       
                                                  
                            

                                                                          
                                                 
                                                                          

                                       
                                   
                       
                                  
                                      
                                 
                                   
                       
                          
               

                                        
                                    
                       
                                      
                                    
                                            
                       
                          
               

                                                   
                             
                       
                                  
                                    
                
                           
                              
           

                                       

    def run(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete time-series power-flow simulation.
        """

        self.load_network()
        self.load_profiles()

        assert self.net is not None
        assert self.profiles is not None

                                                                        
                       
                                                                        

        der_index = self.add_solar_der()

                                                                        
                                          
                                                                        

        initial_p = self.net.load["p_mw"].copy()
        initial_q = self.net.load["q_mvar"].copy()

        self._bus_results = []
        self._line_results = []

                                                                        
                                  
                                                                        

        for timestep in range(len(self.profiles)):

            row = self.profiles.iloc[timestep]

            load_scale = float(row["load_pu"])
            solar_scale = float(row["solar_pu"])

                                                                        
                                 
                                                                        

            self.net.load["p_mw"] = (
                initial_p * load_scale
            )

            self.net.load["q_mvar"] = (
                initial_q * load_scale
            )

                                                                        
                                      
                                                                        

            self.net.sgen.at[
                der_index,
                "p_mw",
            ] = (
                self.solar_capacity_mw
                * solar_scale
            )

                                                                        
                                
                                                                        

            try:

                pp.runpp(
                    self.net,
                    init="results",
                )

            except Exception as exc:

                timestamp = row["timestamp"]

                raise RuntimeError(
                    f"Power flow failed at timestep "
                    f"{timestep} ({timestamp})."
                ) from exc

                                                                        
                                
                                                                        

            self._bus_results.append(
                self.net.res_bus[
                    [
                        "vm_pu",
                        "va_degree",
                        "p_mw",
                        "q_mvar",
                    ]
                ].copy()
            )

                                                                        
                                 
                                                                        

            self._line_results.append(
                self.net.res_line[
                    [
                        "i_from_ka",
                        "i_to_ka",
                        "loading_percent",
                    ]
                ].copy()
            )

        return self.collect_results()

                                                                        
                  
                                                                        

    @staticmethod
    def save_results(
        bus_measurements: pd.DataFrame,
        line_measurements: pd.DataFrame,
        output_directory: str | Path = (
            "data/simulated/measurements"
        ),
    ) -> None:
        """
        Save simulation results.
        """

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        bus_path = (
            output_directory
            / "bus_measurements.csv"
        )

        line_path = (
            output_directory
            / "line_measurements.csv"
        )

        bus_measurements.to_csv(
            bus_path,
            index=False,
        )

        line_measurements.to_csv(
            line_path,
            index=False,
        )

        print(
            f"Bus measurements saved to: {bus_path}"
        )

        print(
            f"Line measurements saved to: {line_path}"
        )


if __name__ == "__main__":

    simulator = TimeSeriesSimulator(
        network_name="ieee33",
        profile_path=(
            "data/simulated/profiles/profiles.csv"
        ),
        der_bus=18,
        solar_capacity_mw=0.5,
    )

    bus_measurements, line_measurements = (
        simulator.run()
    )

    print(
        "\nTime-series simulation completed successfully."
    )

    print("\nBus measurements:")
    print("-" * 60)

    print(
        bus_measurements.head(10)
    )

    print("\nLine measurements:")
    print("-" * 60)

    print(
        line_measurements.head(10)
    )

    simulator.save_results(
        bus_measurements,
        line_measurements,
    )