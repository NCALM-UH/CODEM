# CODEM ArcGIS Pro Integration

The CODEM package can be used within ArcGIS Pro 2.9 through the use of the [toolbox](./src/coregister/esri/toolboxes/3D%20Data%20Co-Registration.pyt).

To utilize this capability, a conda envrionment much be created that ArcGIS will read in, and have all the necessary dependencies installed with.

## Configuration

From the root of the of the CODEM project

```doscon
> conda create --name codem --file .\arcgispro\spec-file.txt
> conda activate codem
> pip install -e .
```

Within ArcGIS, go to the Python tab, and add a newly created conda environment (this can be retrieved by running `conda info | findstr /c:"active env location"` from the command prompt with the `codem` environment activated).
