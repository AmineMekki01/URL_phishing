import React from 'react';
import { Typography } from '@mui/material';
import mibImage from './../assets/mib.png';
import styled from 'styled-components';


const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
`;

const Home = () => {
  return (
    <Container>
      <Typography variant="h4" gutterBottom>
        Welcome to the URL Phishing Detection System
      </Typography>
      <Typography variant="body1">
        Use this system to train models for phishing detection and predict the legitimacy of URLs.
      </Typography>
      <img src={mibImage} alt="mib" style={{ width: '100%', height: 'auto', marginTop: '2rem' }} />
    </Container>
  );
};

export default Home;
