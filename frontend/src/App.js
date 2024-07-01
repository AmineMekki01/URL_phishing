import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import { Container, AppBar, Toolbar, Typography, Button } from '@mui/material';
import Home from './pages/Home';
import TrainingForm from './pages/TrainingForm';
import PredictionForm from './pages/PredictionForm';

const App = () => {
  return (
    <Router>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" style={{ flexGrow: 1 }}>
            URL Phishing Detection
          </Typography>
          <Button color="inherit" component={Link} to="/">
            Home
          </Button>
          <Button color="inherit" component={Link} to="/train">
            Train Model
          </Button>
          <Button color="inherit" component={Link} to="/predict">
            Predict URL
          </Button>
        </Toolbar>
      </AppBar>
      <Container style={{ marginTop: '2rem' }}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/train" element={<TrainingForm />} />
          <Route path="/predict" element={<PredictionForm />} />
        </Routes>
      </Container>
    </Router>
  );
};

export default App;
