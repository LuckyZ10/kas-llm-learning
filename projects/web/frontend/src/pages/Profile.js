import React from 'react';
import { Box, Typography, Paper, Grid, Card, CardContent, Avatar } from '@mui/material';
import { useAuthStore } from '../store/authStore';

const Profile = () => {
  const { user } = useAuthStore();
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>Profile</Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Avatar sx={{ width: 64, height: 64, mr: 2, fontSize: 24, bgcolor: 'primary.main' }}>
                {(user?.full_name?.[0] || user?.username?.[0] || 'U').toUpperCase()}
              </Avatar>
              <Box>
                <Typography variant="h6">{user?.full_name || user?.username}</Typography>
                <Typography color="textSecondary">@{user?.username}</Typography>
              </Box>
            </Box>
            
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Typography variant="subtitle2" color="textSecondary">Email</Typography>
                <Typography>{user?.email}</Typography>
              </Grid>
              
              <Grid item xs={12}>
                <Typography variant="subtitle2" color="textSecondary">Institution</Typography>
                <Typography>{user?.institution || 'Not specified'}</Typography>
              </Grid>
              
              <Grid item xs={12}>
                <Typography variant="subtitle2" color="textSecondary">Role</Typography>
                <Typography sx={{ textTransform: 'capitalize' }}>{user?.role}</Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Account Statistics</Typography>
            
            <Grid container spacing={2}>
              {[
                { label: 'Total Tasks', value: 0 },
                { label: 'Completed', value: 0 },
                { label: 'Member Since', value: new Date(user?.created_at).toLocaleDateString() },
              ].map((stat) => (
                <Grid item xs={6} key={stat.label}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h4" color="primary">{stat.value}</Typography>
                      <Typography variant="body2" color="textSecondary">{stat.label}</Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Profile;
