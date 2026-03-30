import React from 'react';
import { Box, Typography, Paper, Grid, Card, CardContent, LinearProgress } from '@mui/material';

const ResultsDashboard = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>Results Dashboard</Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Performance Metrics</Typography>
            
            <Grid container spacing={3} sx={{ mt: 1 }}>
              {[
                { label: 'Ion Conductivity', value: 0.42, unit: 'mS/cm', max: 1.0 },
                { label: 'Diffusion Coefficient', value: 1.2e-5, unit: 'cm²/s', max: 2e-5 },
                { label: 'Activation Energy', value: 0.25, unit: 'eV', max: 1.0 },
                { label: 'Band Gap', value: 3.2, unit: 'eV', max: 5.0 },
              ].map((metric) => (
                <Grid item xs={12} md={6} lg={3} key={metric.label}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography color="textSecondary" gutterBottom>{metric.label}</Typography>
                      <Typography variant="h4">{metric.value.toExponential ? metric.value.toExponential(2) : metric.value} {metric.unit}</Typography>
                      <LinearProgress
                        variant="determinate"
                        value={(metric.value / metric.max) * 100}
                        sx={{ mt: 1, height: 6, borderRadius: 3 }}
                      />
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>Temperature Dependence</Typography>
            <Typography color="textSecondary">Plot: Arrhenius fit for ionic conductivity</Typography>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>Radial Distribution Function</Typography>
            <Typography color="textSecondary">Plot: RDF from MD trajectory</Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ResultsDashboard;
