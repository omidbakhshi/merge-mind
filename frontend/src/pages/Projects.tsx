import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Grid,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Switch,
  FormControlLabel,
  Alert,
  CircularProgress,
  Fab,
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Folder as FolderIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
} from '@mui/icons-material';
import { apiService, Project } from '../services/api';

const Projects: React.FC = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingProject, setEditingProject] = useState<Project | null>(null);
  const [formData, setFormData] = useState<Partial<Project>>({
    name: '',
    description: '',
    review_enabled: true,
    review_drafts: false,
    min_lines_changed: 10,
    max_files_per_review: 50,
    excluded_paths: ["vendor/", "node_modules/", "dist/", "build/"],
    included_extensions: [".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c"],
  });

  const loadProjects = async () => {
    try {
      setLoading(true);
      const response = await apiService.getProjects();
      setProjects(response.projects || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load projects');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadProjects();
  }, []);

  const handleOpenDialog = (project?: Project) => {
    if (project) {
      setEditingProject(project);
      setFormData({ ...project });
    } else {
      setEditingProject(null);
      setFormData({
        name: '',
        description: '',
        review_enabled: true,
        review_drafts: false,
        min_lines_changed: 10,
        max_files_per_review: 50,
        excluded_paths: ["vendor/", "node_modules/", "dist/", "build/"],
        included_extensions: [".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c"],
      });
    }
    setDialogOpen(true);
  };

  const handleCloseDialog = () => {
    setDialogOpen(false);
    setEditingProject(null);
    setFormData({});
  };

  const handleSave = async () => {
    try {
      if (editingProject) {
        await apiService.updateProject(editingProject.project_id, formData);
      } else {
        await apiService.createProject(formData as Omit<Project, 'project_id'> & { project_id: number });
      }
      await loadProjects();
      handleCloseDialog();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save project');
    }
  };

  const handleDelete = async (projectId: number) => {
    if (window.confirm(`Are you sure you want to delete project ${projectId}?`)) {
      try {
        await apiService.deleteProject(projectId);
        await loadProjects();
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to delete project');
      }
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Projects
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => handleOpenDialog()}
        >
          Add Project
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {projects.map((project) => (
          <Grid item xs={12} md={6} lg={4} key={project.project_id}>
            <Card>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                  <Box>
                    <Typography variant="h6" component="h2">
                      {project.name}
                    </Typography>
                    <Typography color="textSecondary">
                      ID: {project.project_id}
                    </Typography>
                  </Box>
                  <Box>
                    <IconButton
                      size="small"
                      onClick={() => handleOpenDialog(project)}
                    >
                      <EditIcon />
                    </IconButton>
                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => handleDelete(project.project_id)}
                    >
                      <DeleteIcon />
                    </IconButton>
                  </Box>
                </Box>

                <Box mb={2}>
                  <Chip
                    icon={project.review_enabled ? <CheckCircleIcon /> : <CancelIcon />}
                    label={project.review_enabled ? "Reviews Enabled" : "Reviews Disabled"}
                    color={project.review_enabled ? "success" : "default"}
                    size="small"
                  />
                </Box>

                <Typography variant="body2" color="textSecondary" paragraph>
                  Min lines: {project.min_lines_changed} | Max files: {project.max_files_per_review}
                </Typography>

                <Box>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    Extensions: {project.included_extensions?.join(', ') || 'None'}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Excluded: {project.excluded_paths?.join(', ') || 'None'}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {projects.length === 0 && !loading && (
        <Box textAlign="center" py={8}>
          <FolderIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="textSecondary">
            No projects configured
          </Typography>
          <Typography variant="body2" color="textSecondary" mb={3}>
            Add your first project to start reviewing merge requests
          </Typography>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => handleOpenDialog()}
          >
            Add Project
          </Button>
        </Box>
      )}

      {/* Project Dialog */}
      <Dialog open={dialogOpen} onClose={handleCloseDialog} maxWidth="md" fullWidth>
        <DialogTitle>
          {editingProject ? 'Edit Project' : 'Add New Project'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            {!editingProject && (
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Project ID"
                  type="number"
                  value={formData.project_id || ''}
                  onChange={(e) => setFormData({ ...formData, project_id: parseInt(e.target.value) })}
                />
              </Grid>
            )}
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Project Name"
                value={formData.name || ''}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                value={formData.description || ''}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                multiline
                rows={2}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Min Lines Changed"
                type="number"
                value={formData.min_lines_changed || 10}
                onChange={(e) => setFormData({ ...formData, min_lines_changed: parseInt(e.target.value) })}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Max Files per Review"
                type="number"
                value={formData.max_files_per_review || 50}
                onChange={(e) => setFormData({ ...formData, max_files_per_review: parseInt(e.target.value) })}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.review_enabled || false}
                    onChange={(e) => setFormData({ ...formData, review_enabled: e.target.checked })}
                  />
                }
                label="Enable Reviews"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.review_drafts || false}
                    onChange={(e) => setFormData({ ...formData, review_drafts: e.target.checked })}
                  />
                }
                label="Review Draft MRs"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Included Extensions (comma-separated)"
                value={formData.included_extensions?.join(', ') || ''}
                onChange={(e) => setFormData({
                  ...formData,
                  included_extensions: e.target.value.split(',').map(s => s.trim()).filter(s => s)
                })}
                helperText="File extensions to include in reviews (e.g., .py, .js, .ts)"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Excluded Paths (comma-separated)"
                value={formData.excluded_paths?.join(', ') || ''}
                onChange={(e) => setFormData({
                  ...formData,
                  excluded_paths: e.target.value.split(',').map(s => s.trim()).filter(s => s)
                })}
                helperText="Paths to exclude from reviews (e.g., vendor/, node_modules/)"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button onClick={handleSave} variant="contained">
            {editingProject ? 'Update' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Projects;