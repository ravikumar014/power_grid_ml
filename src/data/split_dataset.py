     
                                                             

                                                                  

           
                
          

                                                                      
                                                    

         
         
        

                                

                   

                            
                           
                            

                                                                       
     

                                    

                                   
                          

                     


            
                    
         
                                                        
         

                     

                     
                          
                    

                               
                             

                                    
                                  

                              
                            

                        
                             
                       

                               
             
                                               
             

                   
                 
                       
                                                 
                       
                                                      
                 
                      
                                                      
                                                           
                                                       
                                                     
                 
                           
                                                           
                                                                
                                                            
                                                          
                 
                     
                                                     
                                                          
                                                      
                                                    
                 
                       
           

                                 


                         
         
                                                             

                
                
                  
                                                        

                       
                                          

                 
                                       

                    
                                                             
         

                   
               
                                    
                                         
                                   
                                        
                

                   
                         
                                
                          
           

                                     
                               
                                                     
                                              
               

                 
                          
                               
                         
                 
                               
                                                               
               

                                        
                                                  
                                      

                                            

                                                                          
                  
                                                                          

                
               
                                  
                 
                       
                       
                       
                      
        
             
                                                        

                                                                    
                          

                                                                          
                                   

                 
                 
               
                    
              
                
             

                               
                       
           

                                 

                                             
                                
                             
           

                                                                          
                               
                                                                          

                                  
                            
                                  

                                                                          
                                        
           
                    
           
                                
                                
                                
           
                                                         
                                                                          

                        
                               
                                
                            
                                     
           

                                        

                              
                               
                                                              
                                                        
               

                                                                          
                                          
                                                                          

                                
                          
                                
           

                                  
                             
                    
                              
                                         
               
           

                                                               
                                
                
                              
           

                                     
                                  
                                   
           

                                     
                                   
                               
           

                                        
                              
           

                                             
                                                  
           

                                       
                                   
           

              
                                        
                                                
                                          
            
                               
                                                        
               

                                                                          
                                       
                                                                          

                       
                                     
                                  
               
                  

                            
                                     
                                       
               
                  

                      
                                     
                                 
               
                  

                                                                          
                           
                                                                          

                                   
                   
           

                                        
                        
           

                                  
                  
           

                                                                          
                                       
                                                                          

                                  
                    
                         
                   
           

                                                                          
                          
                                                                          

                               
                                   

                                    
                                              
                                  

                                                   
                                                 

                                
                                               
                
                              
                                               
                

                                                 
                                               

                           
                                        
                
                                
                                             
                
                          
                                       
                
           

                  
                    
                         
                   
                     
           

                                                                          
                        
                                                                          

                          
               
                                  
                

                             
                               
                                                    
               

                              
                          
                                 
           

                     
                              
                                      
           

                     
                               
                                              
                                      
               

                                                 
                               
                                                             
               

                                                                          
            
                                                                          

                      
               
                                  
                        

                                       
                  
                              
                                     
               
                                  

                                                                          
                     
                                                                          

                   
                             
                              
                                   
                             
                

                            
                                         
           

                                 
                                              
           

                           
                                        
           

                                            
                                 
                                                      
                                              
               

                                      
                                 
                                                      
                                        
               

                                           
                                 
                                                      
                                             
               

                                                                          
            
                                                                          

                   
                      
                              
                                   
                             
                                                          
                
             
                                         
             

                                  
                              
           

                                 
                           
                            
           

                       
                                             
                          
           

                            
                                                  
                          
           

                      
                                            
                          
           


                          
                             
                                                      
                                
                   
         
                                                                   
         

                                   

                                 
                                  
                                                
           

                         
                     
                                    
       

                       
                  
                                  
                        
       

                                  
                           
                                
                          
                                      
       

       
                
                     
               
                 
                              

                           
                
                     
               
                           
       

                   


                            

                        
                           
                                          
       

                              
                          
       

                                   
                                
                                            
                             
       

            
                          
       


"""
Chronological dataset splitting for power-grid anomaly detection.

The split is temporal rather than random.

Why?
----
Power-grid measurements are time-series observations. Randomly mixing
future observations into the training set can produce temporal leakage.

Default split:

    60% training
    20% validation
    20% test

The splitter operates using timestamps and preserves the original row
ordering within each partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


                                                                        
               
                                                                        


@dataclass
class TemporalSplitConfig:
    """
    Configuration for chronological train/validation/test splitting.
    """

    train_fraction: float = 0.60

    validation_fraction: float = 0.20

    test_fraction: float = 0.20

    timestamp_column: str = "timestamp"


                                                                        
                   
                                                                        


class TemporalSplitter:
    """
    Chronological train/validation/test splitter.
    """

    def __init__(
        self,
        config: TemporalSplitConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else TemporalSplitConfig()
        )

        self._validate_config()

                                                                        
                              
                                                                        

    def _validate_config(
        self,
    ) -> None:

        fractions = [
            self.config.train_fraction,
            self.config.validation_fraction,
            self.config.test_fraction,
        ]

        if any(
            fraction <= 0
            for fraction in fractions
        ):

            raise ValueError(
                "All split fractions must be > 0."
            )

        total = sum(
            fractions
        )

        if abs(
            total - 1.0
        ) > 1e-8:

            raise ValueError(
                "Train, validation, and test "
                f"fractions must sum to 1.0. "
                f"Got {total}."
            )

                                                                        
                        
                                                                        

    def _prepare(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):

            raise TypeError(
                "Expected pandas DataFrame."
            )

        if dataframe.empty:

            raise ValueError(
                "Cannot split an empty dataframe."
            )

        timestamp_column = (
            self.config.timestamp_column
        )

        if timestamp_column not in (
            dataframe.columns
        ):

            raise ValueError(
                f"Missing timestamp column: "
                f"{timestamp_column}"
            )

        result = dataframe.copy()

        result[
            timestamp_column
        ] = pd.to_datetime(
            result[
                timestamp_column
            ],
            errors="coerce",
        )

        if result[
            timestamp_column
        ].isna().any():

            raise ValueError(
                "Dataset contains invalid timestamps."
            )

                                                                        
                                 
                                                                        

        result = result.sort_values(
            timestamp_column,
            kind="stable",
        ).reset_index(
            drop=True
        )

        return result

                                                                        
           
                                                                        

    def split(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Split dataframe chronologically using UNIQUE timestamps.

        All observations belonging to the same timestamp remain in
        the same partition.

        This is critical for multi-entity time-series data such as
        power grids, where every timestamp contains measurements from
        multiple buses or lines.
        """

        data = self._prepare(
            dataframe
        )

        timestamp_column = (
            self.config.timestamp_column
        )

                                                                        
                                
                                                                        

        unique_timestamps = (
            data[
                timestamp_column
            ]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        n_timestamps = len(
            unique_timestamps
        )

        if n_timestamps < 3:

            raise ValueError(
                "At least three unique timestamps "
                "are required for train/validation/test "
                "splitting."
            )

                                                                        
                                         
                                                                        

        train_end = int(
            n_timestamps
            * self.config.train_fraction
        )

        validation_end = (
            train_end
            + int(
                n_timestamps
                * self.config.validation_fraction
            )
        )

                                                                        
                                         
                                                                        

        train_end = max(
            1,
            train_end,
        )

        validation_end = max(
            train_end + 1,
            validation_end,
        )

        validation_end = min(
            validation_end,
            n_timestamps - 1,
        )

                                                                        
                               
                                                                        

        train_timestamps = (
            unique_timestamps.iloc[
                :train_end
            ]
        )

        validation_timestamps = (
            unique_timestamps.iloc[
                train_end:validation_end
            ]
        )

        test_timestamps = (
            unique_timestamps.iloc[
                validation_end:
            ]
        )

                                                                        
                      
                                                                        

        train_mask = data[
            timestamp_column
        ].isin(
            train_timestamps
        )

        validation_mask = data[
            timestamp_column
        ].isin(
            validation_timestamps
        )

        test_mask = data[
            timestamp_column
        ].isin(
            test_timestamps
        )

                                                                        
                            
                                                                        

        train = data.loc[
            train_mask
        ].copy()

        validation = data.loc[
            validation_mask
        ].copy()

        test = data.loc[
            test_mask
        ].copy()

                                                                        
                        
                                                                        

        train = train.reset_index(
            drop=True
        )

        validation = validation.reset_index(
            drop=True
        )

        test = test.reset_index(
            drop=True
        )

                                                                        
                        
                                                                        

        train_times = set(
            train[
                timestamp_column
            ]
        )

        validation_times = set(
            validation[
                timestamp_column
            ]
        )

        test_times = set(
            test[
                timestamp_column
            ]
        )

        if train_times & validation_times:

            raise RuntimeError(
                "Temporal leakage detected between "
                "train and validation."
            )

        if validation_times & test_times:

            raise RuntimeError(
                "Temporal leakage detected between "
                "validation and test."
            )

        if train_times & test_times:

            raise RuntimeError(
                "Temporal leakage detected between "
                "train and test."
            )

        return (
            train,
            validation,
            test,
        )

                                                                        
             
                                                                        

    def summarize(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate a human-readable split summary.
        """

        timestamp_column = (
            self.config.timestamp_column
        )

        rows = []

        for name, dataframe in [
            ("train", train),
            ("validation", validation),
            ("test", test),
        ]:

            rows.append(
                {
                    "split": name,
                    "rows": len(dataframe),
                    "start": dataframe[
                        timestamp_column
                    ].min(),
                    "end": dataframe[
                        timestamp_column
                    ].max(),
                    "unique_timestamps": (
                        dataframe[
                            timestamp_column
                        ]
                        .nunique()
                    ),
                }
            )

        return pd.DataFrame(
            rows
        )


                                                                        
                                          
                                                                        


def split_aligned_data(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    config: TemporalSplitConfig | None = None,
) -> dict[
    str,
    tuple[
        pd.DataFrame,
        pd.DataFrame,
    ],
]:
    """
    Split feature matrix and metadata using identical timestamp masks.

    Every row in the feature matrix remains aligned with its metadata.

    All observations belonging to the same timestamp are kept in the
    same partition.
    """

    if len(features) != len(
        metadata
    ):

        raise ValueError(
            "Features and metadata have different "
            "numbers of rows."
        )

    if config is None:

        config = TemporalSplitConfig()

    timestamp_column = (
        config.timestamp_column
    )

    if timestamp_column not in metadata.columns:

        raise ValueError(
            f"Metadata missing timestamp column: "
            f"{timestamp_column}"
        )

                                                                    
                                                   
                                                                    

    features = features.reset_index(
        drop=True
    )

    metadata = metadata.reset_index(
        drop=True
    )

    metadata[
        timestamp_column
    ] = pd.to_datetime(
        metadata[
            timestamp_column
        ],
        errors="coerce",
    )

    if metadata[
        timestamp_column
    ].isna().any():

        raise ValueError(
            "Metadata contains invalid timestamps."
        )

                                                                    
                        
                                                                    

    timestamps = (
        metadata[
            timestamp_column
        ]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    n_timestamps = len(
        timestamps
    )

    train_end = int(
        n_timestamps
        * config.train_fraction
    )

    validation_end = (
        train_end
        + int(
            n_timestamps
            * config.validation_fraction
        )
    )

    train_end = max(
        1,
        train_end,
    )

    validation_end = max(
        train_end + 1,
        validation_end,
    )

    validation_end = min(
        validation_end,
        n_timestamps - 1,
    )

    train_timestamps = set(
        timestamps.iloc[
            :train_end
        ]
    )

    validation_timestamps = set(
        timestamps.iloc[
            train_end:validation_end
        ]
    )

    test_timestamps = set(
        timestamps.iloc[
            validation_end:
        ]
    )

                                                                    
            
                                                                    

    train_mask = metadata[
        timestamp_column
    ].isin(
        train_timestamps
    )

    validation_mask = metadata[
        timestamp_column
    ].isin(
        validation_timestamps
    )

    test_mask = metadata[
        timestamp_column
    ].isin(
        test_timestamps
    )

                                                                    
                                                     
                                                                    

    train_features = features.loc[
        train_mask
    ].reset_index(
        drop=True
    )

    train_metadata = metadata.loc[
        train_mask
    ].reset_index(
        drop=True
    )

    validation_features = features.loc[
        validation_mask
    ].reset_index(
        drop=True
    )

    validation_metadata = metadata.loc[
        validation_mask
    ].reset_index(
        drop=True
    )

    test_features = features.loc[
        test_mask
    ].reset_index(
        drop=True
    )

    test_metadata = metadata.loc[
        test_mask
    ].reset_index(
        drop=True
    )

                                                                    
                       
                                                                    

    if len(train_features) != len(
        train_metadata
    ):
        raise RuntimeError(
            "Train feature/metadata alignment failed."
        )

    if len(validation_features) != len(
        validation_metadata
    ):
        raise RuntimeError(
            "Validation feature/metadata alignment failed."
        )

    if len(test_features) != len(
        test_metadata
    ):
        raise RuntimeError(
            "Test feature/metadata alignment failed."
        )

                                                                    
             
                                                                    

    return {
        "train": (
            train_features,
            train_metadata,
        ),

        "validation": (
            validation_features,
            validation_metadata,
        ),

        "test": (
            test_features,
            test_metadata,
        ),
    }


                                                                        
      
                                                                        


if __name__ == "__main__":

    print()
    print(
        "=" * 70
    )

    print(
        "TEMPORAL DATASET SPLIT"
    )

    print(
        "=" * 70
    )

                                                                    
                   
                                                                    

    config = TemporalSplitConfig(
        train_fraction=0.60,
        validation_fraction=0.20,
        test_fraction=0.20,
    )

    splitter = TemporalSplitter(
        config
    )

                                                                    
         
                                                                    

    print()
    print(
        "-" * 70
    )

    print(
        "BUS DATA"
    )

    print(
        "-" * 70
    )

    bus_features = pd.read_csv(
        "data/features/final/"
        "bus_feature_matrix.csv"
    )

    bus_metadata = pd.read_csv(
        "data/features/final/"
        "bus_metadata.csv"
    )

    bus_splits = split_aligned_data(
        bus_features,
        bus_metadata,
        config,
    )

    for name, (
        features,
        metadata,
    ) in bus_splits.items():

        print(
            f"{name:12s} "
            f"features={features.shape} "
            f"rows={len(metadata)} "
            f"start={metadata['timestamp'].iloc[0]} "
            f"end={metadata['timestamp'].iloc[-1]}"
        )

                                                                    
          
                                                                    

    print()
    print(
        "-" * 70
    )

    print(
        "LINE DATA"
    )

    print(
        "-" * 70
    )

    line_features = pd.read_csv(
        "data/features/final/"
        "line_feature_matrix.csv"
    )

    line_metadata = pd.read_csv(
        "data/features/final/"
        "line_metadata.csv"
    )

    line_splits = split_aligned_data(
        line_features,
        line_metadata,
        config,
    )

    for name, (
        features,
        metadata,
    ) in line_splits.items():

        print(
            f"{name:12s} "
            f"features={features.shape} "
            f"rows={len(metadata)} "
            f"start={metadata['timestamp'].iloc[0]} "
            f"end={metadata['timestamp'].iloc[-1]}"
        )

    print()
    print(
        "=" * 70
    )

    print(
        "TEMPORAL SPLIT COMPLETE"
    )

    print(
        "=" * 70
    )