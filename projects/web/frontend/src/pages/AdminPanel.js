import React from 'react';
import { Box, Typography, Paper, Grid, Card, CardContent, Tabs, Tab } from '@mui/material';

const AdminPanel = () => {
  const [activeTab, setActiveTab] = React.useState(0);
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>Admin Panel</Typography>
      
      <Paper>
        <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)} sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tab label="Users" />
          <Tab label="Queue Status" />
          <Tab label="System Stats" />
          <Tab label="Settings" />
        </Tabs>
        
        <Box sx={{ p: 3 }}>
          {activeTab === 0 && <Typography>User management interface</Typography>}
          {activeTab === 1 && <Typography>Task queue status and monitoring</Typography>}
          {activeTab === 2 && <Typography>System statistics and health</Typography>}
          {activeTab === 3 && <Typography>System configuration settings</Typography>}
        </Box>
      </Paper>
    </Box>
  );
};

export default AdminPanel;
