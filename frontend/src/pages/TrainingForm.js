import React, { useState } from 'react';
import axios from 'axios';
import {
  TextField, Button, MenuItem, FormControlLabel, Checkbox, Typography, Table,
  TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, FormControl, InputLabel, Select
} from '@mui/material';

const hyperparameterOptions = {
  rf: {
    numTrees: { step: 50 },
    maxDepth: { step: 5 },
    maxBins: { step: 32 },
    minInstancesPerNode: { step: 1 },
    subsamplingRate: { step: 0.1 },
    featureSubsetStrategy: { options: ['auto', 'sqrt', 'log2'] }
  },
  gbt: {
    maxIter: { step: 10 },
    maxDepth: { step: 2 },
    maxBins: { step: 32 },
    minInstancesPerNode: { step: 1 },
    stepSize: { step: 0.05 },
    subsamplingRate: { step: 0.1 }
  },
  lr: {
    regParam: { step: 0.1 },
    elasticNetParam: { step: 0.1 },
    maxIter: { step: 10 },
    fitIntercept: { options: [true, false] },
    tol: { step: 1e-4 }
  }
};

const TrainingForm = () => {
  const [modelName, setModelName] = useState('rf');
  const [searchType, setSearchType] = useState('default');
  const [filePath, setFilePath] = useState('artifacts/data/processed/urls_dataset.parquet');
  const [isProcessed, setIsProcessed] = useState(false);
  const [isStacking, setIsStacking] = useState(false);
  const [baseModelNames, setBaseModelNames] = useState('rf, gbt, lr');
  const [metaModelName, setMetaModelName] = useState('lr');
  const [userParams, setUserParams] = useState({});
  const [results, setResults] = useState(null);

  const handleModelChange = (e) => {
    setModelName(e.target.value);
    setUserParams({});
  };

  const handleParamChange = (e) => {
    const { name, value } = e.target;
    setUserParams({
      ...userParams,
      [name]: value.split(',').map(val => val.trim())
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const payload = {
      file_path: filePath,
      model_name: modelName,
      search_type: searchType,
      user_params: userParams,
      is_processed: isProcessed,
      is_stacking: isStacking,
      base_model_names: baseModelNames.split(',').map(name => name.trim()),
      meta_model_name: metaModelName
    };

    try {
      const response = await axios.post('http://localhost:8000/train', payload);
      console.log(response.data);
      setResults(response.data);
      alert('Training initiated successfully');
    } catch (error) {
      console.error(error);
      alert('Error initiating training');
    }
  };

  const renderHyperparameters = () => {
    const options = hyperparameterOptions[modelName] || {};
    return Object.keys(options).map(param => {
      const paramOptions = options[param];
      if (paramOptions.options) {
        return (
          <FormControl fullWidth margin="normal" key={param}>
            <InputLabel>{param}</InputLabel>
            <Select
              name={param}
              value={userParams[param] ? userParams[param].join(',') : ''}
              onChange={handleParamChange}
            >
              {paramOptions.options.map(option => (
                <MenuItem key={option} value={option}>
                  {option.toString()}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        );
      } else {
        return (
          <TextField
            label={param}
            name={param}
            type="text"
            value={userParams[param] ? userParams[param].join(',') : ''}
            onChange={handleParamChange}
            fullWidth
            margin="normal"
            key={param}
            placeholder={`Enter values separated by commas (step: ${paramOptions.step})`}
          />
        );
      }
    });
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <Typography variant="h5" gutterBottom>
          Train Model
        </Typography>
        <TextField
          label="File Path"
          value={filePath}
          onChange={(e) => setFilePath(e.target.value)}
          fullWidth
          margin="normal"
        />

        

        <FormControlLabel
          control={<Checkbox checked={isProcessed} onChange={(e) => setIsProcessed(e.target.checked)} />}
          label="Is Processed"
        />
        <FormControlLabel
          control={<Checkbox checked={isStacking} onChange={(e) => setIsStacking(e.target.checked)} />}
          label="Use Stacking"
        />
        {!isStacking && (
          <>
            <TextField
              select
              label="Model Name"
              value={modelName}
              onChange={handleModelChange}
              fullWidth
              margin="normal"
            >
              <MenuItem value="rf">Random Forest</MenuItem>
              <MenuItem value="gbt">Gradient Boosting</MenuItem>
              <MenuItem value="lr">Logistic Regression</MenuItem>
            </TextField>
            {searchType !== 'default' && renderHyperparameters()}
            <TextField
              select
              label="Search Type"
              value={searchType}
              onChange={(e) => setSearchType(e.target.value)}
              fullWidth
              margin="normal"
            >
              <MenuItem value="default">Default</MenuItem>
              <MenuItem value="grid">Grid Search</MenuItem>
              <MenuItem value="random">Random Search</MenuItem>
            </TextField>
          </>
          
        )}
        {isStacking && (
          <>
            <TextField
              label="Base Model Names (comma separated)"
              value={baseModelNames}
              onChange={(e) => setBaseModelNames(e.target.value)}
              fullWidth
              margin="normal"
            />
            <TextField
              label="Meta Model Name"
              value={metaModelName}
              onChange={(e) => setMetaModelName(e.target.value)}
              fullWidth
              margin="normal"
            />
          </>
        )}
        <Button type="submit" variant="contained" color="primary" style={{ marginTop: '1rem' }}>
          Start Training
        </Button>
      </form>
      {results && (
        <TableContainer component={Paper} style={{ marginTop: '2rem' }}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Metric</TableCell>
                <TableCell>Value</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              <TableRow>
                <TableCell>Accuracy</TableCell>
                <TableCell>{results.accuracy}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Precision</TableCell>
                <TableCell>{results.precision}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Recall</TableCell>
                <TableCell>{results.recall}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>F1 Score</TableCell>
                <TableCell>{results.f1}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>AUC</TableCell>
                <TableCell>{results.auc}</TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </div>
  );
};

export default TrainingForm;
