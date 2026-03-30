import React, { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Button,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Alert,
  IconButton,
  Tooltip,
  Tab,
  Tabs,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Cancel as CancelIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Schedule as ScheduleIcon,
  PlayArrow as PlayArrowIcon,
} from '@mui/icons-material';
import { useTaskStore } from '../store/authStore';

const TaskDetail = () => {
  const { taskId } = useParams();
  const navigate = useNavigate();
  const { currentTask, fetchTask, cancelTask, isLoading, error } = useTaskStore();
  const [activeTab, setActiveTab] = React.useState(0);
  
  useEffect(() => {
    fetchTask(taskId);
    
    // Poll for updates if task is running
    const interval = setInterval(() => {
      if (currentTask?.status === 'running') {
        fetchTask(taskId);
      }
    }, 5000);
    
    return () => clearInterval(interval);
  }, [taskId, fetchTask]);
  
  const handleCancel = async () => {
    await cancelTask(taskId);
    fetchTask(taskId);
  };
  
  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircleIcon color="success" />;
      case 'failed':
      case 'cancelled':
        return <ErrorIcon color="error" />;
      case 'running':
        return <PlayArrowIcon color="primary" />;
      default:
        return <ScheduleIcon color="disabled" />;
    }
  };
  
  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'failed':
      case 'cancelled':
        return 'error';
      case 'running':
        return 'primary';
      case 'pending':
      case 'queued':
        return 'warning';
      default:
        return 'default';
    }
  };
  
  if (!currentTask && isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <Typography>Loading task details...</Typography>
      </Box>
    );
  }
  
  if (!currentTask) {
    return (
      <Box>
        <Alert severity="error">Task not found</Alert>
        <Button sx={{ mt: 2 }} onClick={() => navigate('/tasks')}>Back to Tasks</Button>
      </Box>
    );
  }
  
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">{currentTask.name}</Typography>
        <Box>
          <Tooltip title="Refresh">
            <IconButton onClick={() => fetchTask(taskId)} disabled={isLoading}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
          
          {['pending', 'queued', 'running'].includes(currentTask.status) && (
            <Button
              variant="outlined"
              color="error"
              startIcon={<CancelIcon />}
              onClick={handleCancel}
              sx={{ ml: 1 }}
            >
              Cancel
            </Button>
          )}
          
          {currentTask.status === 'completed' && (
            <Button
              variant="contained"
              startIcon={<DownloadIcon />}
              sx={{ ml: 1 }}
            >
              Download Results
            </Button>
          )}
        </Box>
      </Box>
      
      {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {getStatusIcon(currentTask.status)}
                <Typography variant="h6">Status: {currentTask.status}</Typography>
              </Box>
              <Chip
                label={currentTask.workflow_type}
                color="primary"
                variant="outlined"
              />
            </Box>
            
            {currentTask.status === 'running' && (
              <Box sx={{ mb: 3 }}>
                <LinearProgress
                  variant="determinate"
                  value={currentTask.progress}
                  sx={{ height: 10, borderRadius: 5 }}
                />
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                  <Typography variant="body2" color="textSecondary">{currentTask.stage}</Typography>
                  <Typography variant="body2" fontWeight="bold">{currentTask.progress.toFixed(1)}%</Typography>
                </Box>
                <Typography variant="body2" color="textSecondary">{currentTask.message}</Typography>
              </Box>
            )}
            
            <Tabs
              value={activeTab}
              onChange={(e, v) => setActiveTab(v)}
              sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}
            >
              <Tab label="Details" />
              <Tab label="Logs" />
              <Tab label="Results" />
            </Tabs>
            
            {activeTab === 0 && (
              <Box>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="subtitle2" color="textSecondary">Created</Typography>
                    <Typography>{new Date(currentTask.created_at).toLocaleString()}</Typography>
                  </Grid>
                  
                  <Grid item xs={6}>
                    <Typography variant="subtitle2" color="textSecondary">Priority</Typography>
                    <Typography>{currentTask.priority}</Typography>
                  </Grid>
                  
                  {currentTask.started_at && (
                    <Grid item xs={6}>
                      <Typography variant="subtitle2" color="textSecondary">Started</Typography>
                      <Typography>{new Date(currentTask.started_at).toLocaleString()}</Typography>
                    </Grid>
                  )}
                  
                  {currentTask.runtime_seconds && (
                    <Grid item xs={6}>
                      <Typography variant="subtitle2" color="textSecondary">Runtime</Typography>
                      <Typography>{Math.round(currentTask.runtime_seconds / 60)} minutes</Typography>
                    </Grid>
                  )}
                </Grid>
                
                {currentTask.description && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2" color="textSecondary">Description</Typography>
                    <Typography>{currentTask.description}</Typography>
                  </Box>
                )}
              </Box>
            )}
            
            {activeTab === 1 && (
              <Box sx={{ bgcolor: '#1a1a1a', color: '#fff', p: 2, borderRadius: 1, fontFamily: 'monospace', fontSize: '0.875rem', maxHeight: 400, overflow: 'auto' }}>
                {currentTask.logs?.map((log, i) => (
                  <Box key={i} sx={{ mb: 0.5 }}>{log}</Box>
                )) || <Typography color="textSecondary">No logs available</Typography>}
              </Box>
            )}
            
            {activeTab === 2 && currentTask.results && (
              <Box>
                <Typography variant="h6" gutterBottom>Results</Typography>
                <pre>{JSON.stringify(currentTask.results, null, 2)}</pre>
              </Box>
            )}
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Configuration</Typography>
            <Divider sx={{ mb: 2 }} />
            
            {currentTask.config && (
              <Box sx={{ fontSize: '0.875rem' }}>
                <Typography variant="subtitle2">DFT Settings:</Typography>
                <Box component="pre" sx={{ bgcolor: 'grey.100', p: 1, borderRadius: 1, overflow: 'auto' }}>{JSON.stringify(currentTask.config.dft_config, null, 2)}</Box>
                
                <Typography variant="subtitle2" sx={{ mt: 2 }}>ML Settings:</Typography>
                <Box component="pre" sx={{ bgcolor: 'grey.100', p: 1, borderRadius: 1, overflow: 'auto' }}>{JSON.stringify(currentTask.config.ml_config, null, 2)}</Box>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default TaskDetail;
