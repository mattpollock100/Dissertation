#model params

TRACE_Temp = {'sub_path' : '/TRACE/', 
                'file' : 'TRACE_TAS.nc', 
                'variable_name' : 'TREFHT',
                'conversion_factor' : 1,
                'y_min' : 290,
                'y_max' : 305,
                'convert_dates' : 2,
                'model_end_year' : 1950}

TRACE_Precip = {'sub_path' : '/TRACE/', 
                'file' : 'TRACE_PRECIP_Final.nc', 
                'variable_name' : 'PRECIP',
                'conversion_factor' : 86400000,
                'y_min' : 0,
                'y_max' : 10,
                'convert_dates' : 2,
                'model_end_year' : 1950}

TRACE_PSL = {'sub_path' : '/TRACE/', 
                'file' : 'TRACE_PSL.nc', 
                'variable_name' : 'PSL',
                'conversion_factor' : 1,
                'y_min' : 0,
                'y_max' : 10,
                'convert_dates' : 2,
                'model_end_year' : 1950}

TRACE_TS = {'sub_path' : '/TRACE/', 
                'file' : 'TRACE_TS.nc', 
                'variable_name' : 'TS',
                'conversion_factor' : 1,
                'y_min' : 290,
                'y_max' : 305,
                'convert_dates' : 2,
                'model_end_year' : 1950}

IPSL_CM6_Precip = {'sub_path' : '/IPSL_CM6/', 
                'file' : 'TR6AV-Sr02_20000101_79991231_1M_precip.nc', 
                'variable_name' : 'precip',
                'conversion_factor' : 86400,
                'y_min' : 0,
                'y_max' : 10,
                'convert_dates' : 2,
                'model_end_year' : 1990}

IPSL_CM6_Temp = {'sub_path' : '/IPSL_CM6/', 
                'file' : 'TR6AV-Sr02_20000101_79991231_1M_t2m.nc', 
                'variable_name' : 'tas',
                'conversion_factor' : 1,
                'y_min' : 295,
                'y_max' : 300,
                'convert_dates' : 2,
                'model_end_year' : 1990}

#https://www.nature.com/articles/s41467-020-18478-6#Sec13
MPI_ESM_Precip = {'sub_path' : '/MPI_ESM/',
                'file' : 'pr_Amon_MPI_ESM_TRSF_slo0043_100101_885012.nc',
                'variable_name' : 'pr',
                'conversion_factor' : 86400,
                'y_min' : 0,
                'y_max' : 10,
                'convert_dates' : 2,
                'model_end_year' : 1850}

MPI_ESM_Temp =   {'sub_path' : '/MPI_ESM/',
                'file' : 'tas_Amon_MPI_ESM_TRSF_slo0043_100101_885012.nc',
                'variable_name' : 'tas',
                'conversion_factor' : 1,
                'y_min' : 295,
                'y_max' : 300,
                'convert_dates' : 2,
                'model_end_year' : 1850}

MPI_ESM_SST =    {'sub_path' : '/MPI_ESM/',
                'file' : 'sst_Amon_MPI_ESM_TRSF_slo0043_100101_885012.nc',
                'variable_name' : 'sst',
                'conversion_factor' : 1,
                'y_min' : 290,
                'y_max' : 305,
                'convert_dates' : 2,
                'model_end_year' : 1850}

MPI_ESM_PSL =    {'sub_path' : '/MPI_ESM/',
                'file' : 'psl_Amon_MPI_ESM_TRSF_slo0043_100101_885012.nc',
                'variable_name' : 'psl',
                'conversion_factor' : 1,
                'y_min' : 98000,
                'y_max' : 105000,
                'convert_dates' : 2,
                'model_end_year' : 1850}

IPSL_CM5_Precip =  {'sub_path' : '/IPSL_CM5/',
                    'file' : 'pr_Amon_TR5AS_combined.nc',
                    'variable_name' : 'pr',
                    'conversion_factor' : 86400,
                    'y_min' : 0,
                    'y_max' : 10,
                    'convert_dates' : 2,
                'model_end_year' : 1990}

IPSL_CM5_Temp =    {'sub_path' : '/IPSL_CM5/',
                    'file' : 'tas_Amon_TR5AS_combined.nc',
                    'variable_name' : 'tas',
                    'conversion_factor' : 1,
                    'y_min' : 295,
                    'y_max' : 300,
                    'convert_dates' : 2,
                    'model_end_year' : 1990}

IPSL_CM5_PSL =    {'sub_path' : '/IPSL_CM5/',
                    'file' : 'psl_Amon_TR5AS_combined.nc',
                    'variable_name' : 'psl',
                    'conversion_factor' : 1,
                    'y_min' : 295,
                    'y_max' : 300,
                    'convert_dates' : 2,
                'model_end_year' : 1990}



all_models = [IPSL_CM5_Precip, IPSL_CM5_Temp, 
            MPI_ESM_Precip, MPI_ESM_Temp, MPI_ESM_PSL, MPI_ESM_SST,
            IPSL_CM6_Precip, IPSL_CM6_Temp, 
            TRACE_Temp, TRACE_Precip, TRACE_PSL]