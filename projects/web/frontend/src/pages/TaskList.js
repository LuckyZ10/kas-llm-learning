import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Chip,
  IconButton,
  TextField,
  InputAdornment,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  Search as SearchIcon,
  Visibility as ViewIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Add as AddIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Schedule as PendingIcon,
  PlayArrow as RunningIcon,
} from '@mui/icons-material';
import { useTaskStore } from '../store/authStore';

const StatusChip = ({ status }) => {
  const config = {
    pending: { color: 'warning', icon: <PendingIcon fontSize="small" /> },
    queued: { color: 'info', icon: <PendingIcon fontSize="small" /> },
    running: { color: 'primary', icon: <RunningIcon fontSize="small" /> },
    completed: { color: 'success', icon: <SuccessIcon fontSize="small" /> },
    failed: { color: 'error', icon: <ErrorIcon fontSize="small" /> },
    cancelled: { color: 'default', icon: <ErrorIcon fontSize="small" /> },
  }[status] || { color: 'default', icon: null };
  
  return (
    <Chip
      icon={config.icon}
      label={status}
      color={config.color}
      size="small"
    />
  );
};

const TaskList = () => {
  const navigate = useNavigate();
  const { tasks, fetchTasks, deleteTask, isLoading, totalTasks } = useTaskStore();
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  
  useEffect(() => {
    fetchTasks({
      skip: page * rowsPerPage,
      limit: rowsPerPage,
      status: statusFilter || undefined,
    });
  }, [page, rowsPerPage, statusFilter, fetchTasks]);
  
  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };
  
  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };
  
  const handleDelete = async (taskId) => {
    if (window.confirm('Are you sure you want to delete this task?')) {
      await deleteTask(taskId);
      fetchTasks({ skip: page * rowsPerPage, limit: rowsPerPage });
    }
  };
  
  const filteredTasks = tasks.filter((task) =>
    task.name.toLowerCase().includes(searchTerm.toLowerCase())
  );
  
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">My Tasks</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => navigate('/submit')}
        >
          New Task
        </Button>
      </Box>
      
      <Paper sx={{ mb: 2, p: 2 }}>
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <TextField
            placeholder="Search tasks..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
            }}
            sx={{ minWidth: 300 }}
          />
          
          <FormControl sx={{ minWidth: 150 }}>
            <InputLabel>Status</InputLabel>
            <Select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              label="Status"
            >
              <MenuItem value="">All</MenuItem>
              <MenuItem value="pending">Pending</MenuItem>
              <MenuItem value="running">Running</MenuItem>
              <MenuItem value="completed">Completed</MenuItem>
              <MenuItem value="failed">Failed</MenuItem>
            </Select>
          </FormControl>
          
          <IconButton onClick={() => fetchTasks()} disabled={isLoading}>
            <RefreshIcon />
          </IconButton>
        </Box>
      </Paper>
      
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Progress</TableCell>
              <TableCell>Created</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredTasks.map((task) => (
              <TableRow key={task.task_id} hover>
                <TableCell>{task.name}</TableCell>
                <TableCell>
                  <Chip label={task.workflow_type} size="small" variant="outlined" />
                </TableCell>
                <TableCell>
                  <StatusChip status={task.status} />
                </TableCell>
                <TableCell>
                  {task.status === 'running' ? `${task.progress.toFixed(0)}%` : '- '}
                </TableCell>
                <TableCell>{new Date(task.created_at).toLocaleDateString()}</TableCell>
                <TableCell>
                  <IconButton onClick={() => navigate(`/tasks/${task.task_id}`)}>
                    <ViewIcon />
                  </IconButton>
                  <IconButton onClick={() => handleDelete(task.task_id)} color="error">
                    <DeleteIcon />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        
        <TablePagination
          component="div"
          count={totalTasks}
          page={page}
          onPageChange={handleChangePage}
          rowsPerPage={rowsPerPage}
          onRowsPerPageChange={handleChangeRowsPerPage}
          rowsPerPageOptions={[5, 10, 25, 50]}
        />
      </TableContainer>
    </Box>
  );
};

export default TaskList;
