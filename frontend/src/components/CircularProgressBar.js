import React from 'react';
import styled from 'styled-components';
import { CircularProgress, Box, Typography } from '@mui/material';

const getGradientColor = (percentage) => {
  const startColor = [255, 87, 51]; // Red color (RGB)
  const endColor = [33, 150, 243]; // Blue color (RGB)
  const diffColor = endColor.map((end, index) => end - startColor[index]);

  const resultColor = startColor.map((start, index) => Math.round(start + diffColor[index] * (percentage / 100)));
  return `rgb(${resultColor.join(',')})`;
};

const CircularProgressBarWrapper = styled(Box)`
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
`;

const CircularProgressBar = ({ percentage }) => {
  const gradientColor = getGradientColor(percentage);

  return (
    <CircularProgressBarWrapper>
      <CircularProgress
        variant="determinate"
        value={percentage}
        size={150}
        thickness={4}
        style={{
          color: gradientColor,
        }}
      />
      <Box
        top={0}
        left={0}
        bottom={0}
        right={0}
        position="absolute"
        display="flex"
        alignItems="center"
        justifyContent="center"
      >
        <Typography variant="h5" component="div" color="textSecondary">
          {`${percentage.toFixed(2)}%`}
        </Typography>
      </Box>
    </CircularProgressBarWrapper>
  );
};

export default CircularProgressBar;
