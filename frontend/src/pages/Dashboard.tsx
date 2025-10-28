import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Alert,
  CircularProgress,
  Chip,
} from '@mui/material';
import {
  Assessment as AssessmentIcon,
  Folder as FolderIcon,
  Code as CodeIcon,
  Timeline as TimelineIcon,
} from '@mui/icons-material';
import { apiService, HealthCheck, Stats } from '../services/api';

const Dashboard: React.FC = () => {
  const [health, setHealth] = useState<HealthCheck | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        const [healthData, statsData] = await Promise.all([
          apiService.getHealth(),
          apiService.getStats(),
        ]);
        setHealth(healthData);
        setStats(statsData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load dashboard data');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  const StatCard: React.FC<{
    title: string;
    value: string | number;
    icon: React.ReactNode;
    color?: string;
  }> = ({ title, value, icon, color = 'primary' }) => (
    <Card>
      <CardContent>
        <Box display="flex" alignItems="center" mb={1}>
          <Box color={`${color}.main`} mr={1}>
            {icon}
          </Box>
          <Typography color="textSecondary" variant="h6">
            {title}
          </Typography>
        </Box>
        <Typography variant="h4" component="div">
          {value}
        </Typography>
      </CardContent>
    </Card>
  );

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Dashboard
      </Typography>

      {health && (
        <Box mb={3}>
          <Alert
            severity={health.status === 'healthy' ? 'success' : 'warning'}
            sx={{ mb: 2 }}
          >
            Service Status: {health.status.toUpperCase()}
            {health.version && ` | Version: ${health.version}`}
          </Alert>
        </Box>
      )}

      <Grid container spacing={3}>
        {stats && (
          <>
            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Projects"
                value={stats.projects_configured}
                icon={<FolderIcon />}
                color="primary"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Active Reviews"
                value={stats.active_reviews}
                icon={<AssessmentIcon />}
                color="secondary"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Cache Size"
                value={stats.cache_size}
                icon={<CodeIcon />}
                color="info"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Learning Operations"
                value={Object.keys(stats.learning_stats || {}).length}
                icon={<TimelineIcon />}
                color="success"
              />
            </Grid>
          </>
        )}
      </Grid>

      {stats?.learning_stats && Object.keys(stats.learning_stats).length > 0 && (
        <Box mt={4}>
          <Typography variant="h5" component="h2" gutterBottom>
            Learning Statistics
          </Typography>
          <Grid container spacing={2}>
            {Object.entries(stats.learning_stats).map(([project, stats]) => (
              <Grid item xs={12} md={6} key={project}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {project}
                    </Typography>
                    <Box display="flex" gap={1} flexWrap="wrap">
                      <Chip
                        label={`Files: ${(stats as any).files_processed || 0}`}
                        size="small"
                        color="primary"
                      />
                      <Chip
                        label={`Chunks: ${(stats as any).total_chunks || 0}`}
                        size="small"
                        color="secondary"
                      />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}
    </Box>
  );
};

export default Dashboard;