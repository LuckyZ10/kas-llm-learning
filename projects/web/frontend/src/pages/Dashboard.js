import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  LinearProgress,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Add as AddIcon,
  Visibility as ViewIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  HourglassEmpty as PendingIcon,
  PlayArrow as RunningIcon,
} from '@mui/icons-material';
import { useTaskStore } from '../store/authStore';

const StatusChip = ({ status }) => {
  const statusConfig = {
    pending: { color: 'warning', icon: <PendingIcon />, label: 'Pending' },
    queued: { color: 'info', icon: <PendingIcon />, label: 'Queued' },
    running: { color: 'primary', icon: <RunningIcon />, label: 'Running' },
    completed: { color: 'success', icon: <SuccessIcon />, label: 'Completed' },
    failed: { color: 'error', icon: <ErrorIcon />, label: 'Failed' },
    cancelled: { color: 'default', icon: <ErrorIcon />, label: 'Cancelled' },
  };
  
  const config = statusConfig[status] || statusConfig.pending;
  
  return (
    <Chip
      icon={config.icon}
      label={config.label}
      color={config.color}
      size="small"
      sx={{ fontWeight: 500 }}
    />
  );
};

const Dashboard = () => {
  const navigate = useNavigate();
  const { tasks, fetchTasks, isLoading } = useTaskStore();
  
  useEffect(() => {
    fetchTasks({ limit: 5 });
    
    // Refresh every 30 seconds
    const interval = setInterval(() => {
      fetchTasks({ limit: 5 });
    }, 30000);
    
    return () => clearInterval(interval);
  }, [fetchTasks]);
  
  const recentTasks = tasks.slice(0, 5);
  
  const stats = {
    total: tasks.length,
    running: tasks.filter((t) => t.status === 'running').length,
    completed: tasks.filter((t) => t.status === 'completed').length,
    failed: tasks.filter((t) => t.status === 'failed').length,
  };
  
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      
      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {[
          { label: 'Total Tasks', value: stats.total, color: 'primary' },
          { label: 'Running', value: stats.running, color: 'info' },
          { label: 'Completed', value: stats.completed, color: 'success' },
          { label: 'Failed', value: stats.failed, color: 'error' },
        ].map((stat) => (
          <Grid item xs={12} sm={6} md={3} key={stat.label}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography color="textSecondary" gutterBottom>
                  {stat.label}
                </Typography>
                <Typography variant="h3" color={`${stat.color}.main`}>
                  {stat.value}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
      
      {/* Recent Tasks */}
      <Paper sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Recent Tasks</Typography>
          <Box>
            <Tooltip title="Refresh">
              <IconButton onClick={() => fetchTasks({ limit: 5 })} disabled={isLoading}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => navigate('/submit')}
              sx={{ ml: 1 }}
            >
              New Task
            </Button>
          </Box>
        </Box>
        
        {recentTasks.length > 0 ? (
          recentTasks.map((task) => (
            <Card key={task.task_id} variant="outlined" sx={{ mb: 2 }} className="task-card">
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                  <Box>
                    <Typography variant="h6" gutterBottom>{task.name}</Typography>
                    <Typography variant="body2" color="textSecondary">
                      Type: {task.workflow_type} • Created: {new Date(task.created_at).toLocaleString()}
                    </Typography>
                  </Box>
                  <StatusChip status={task.status} />
                </Box>
                
                {task.status === 'running' && (
                  <Box sx={{ mt: 2 }}>
                    <LinearProgress
                      variant="determinate"
                      value={task.progress}
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                    <Typography variant="body2" color="textSecondary" sx={{ mt: 0.5 }}>
                      {task.progress.toFixed(1)}% - {task.message}
                    </Typography>
                  </Box>
                )}
              </CardContent>
              <CardActions>
                <Button
                  size="small"
                  startIcon={<ViewIcon />}
                  onClick={() => navigate(`/tasks/${task.task_id}`)}
                >
                  View Details
                </Button>
              </CardActions>
            </Card>
          ))
        ) : (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography color="textSecondary">No tasks yet. Start by creating a new simulation task.</Typography>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => navigate('/submit')}
              sx={{ mt: 2 }}
            >
              Create First Task
            </Button>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default Dashboard;
