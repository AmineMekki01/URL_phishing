import React, { useState } from 'react';
import axios from 'axios';
import { TextField, Button, MenuItem, FormControlLabel, Checkbox, Typography, Box } from '@mui/material';
import CircularProgressBar from '../components/CircularProgressBar';

const PredictionForm = () => {
  const [modelName, setModelName] = useState('rf');
  const [url, setUrl] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [probability, setProbability] = useState(null);
  const [isStacking, setIsStacking] = useState(false);
  const [baseModelNames, setBaseModelNames] = useState('rf, gbt, lr');
  const [metaModelName, setMetaModelName] = useState('lr');

  const handleSubmit = async (e) => {
    e.preventDefault();
    const payload = {
      is_stacking: isStacking,
      base_model_names: baseModelNames.split(',').map(name => name.trim()),
      meta_model_name: metaModelName,
      model_name: modelName,
      url: url,
    };

    try {
      const response = await axios.post('http://localhost:8000/predict', payload, {
        headers: {
          'Content-Type': 'application/json',
        },
      });
      setPrediction(response.data.prediction);
      setProbability(response.data.probability);
    } catch (error) {
      console.error(error);
      alert('Error making prediction');
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <Typography variant="h5" gutterBottom>
          Predict URL
        </Typography>
        
        <TextField
          label="URL"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          fullWidth
          margin="normal"
        />
        <FormControlLabel
          control={<Checkbox checked={isStacking} onChange={(e) => setIsStacking(e.target.checked)} />}
          label="Use Stacking"
        />
        {!isStacking && (
          <TextField
            select
            label="Model Name"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            fullWidth
            margin="normal"
          >
            <MenuItem value="rf">Random Forest</MenuItem>
            <MenuItem value="gbt">Gradient Boosting</MenuItem>
            <MenuItem value="lr">Logistic Regression</MenuItem>
          </TextField>
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
          Predict
        </Button>
      </form>
      {prediction !== null && (
        <div>
          <Typography variant="h6" style={{ marginTop: '1rem' }}>
            Prediction: {prediction === 1 ? 'Phishing' : 'Legitimate'}
          </Typography>
          <Typography variant="h6" style={{ marginTop: '1rem' }}>
            Probability: {(probability * 100).toFixed(2)}%
          </Typography>
          <CircularProgressBar percentage={probability * 100} />
        </div>
      )}
    </div>
  );
};

export default PredictionForm;
