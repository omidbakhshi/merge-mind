import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  TextField,
  Alert,
  Grid,
  Divider,
} from '@mui/material';
import { Refresh as RefreshIcon } from '@mui/icons-material';
import { apiService, ReloadResponse } from '../services/api';

const Settings: React.FC = () => {
  const [reloadResult, setReloadResult] = useState<ReloadResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleReload = async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await apiService.reloadConfig();
      setReloadResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reload configuration');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Settings
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Configuration Reload
              </Typography>
              <Typography variant="body2" color="textSecondary" paragraph>
                Reload configuration from environment variables. This allows you to update
                settings like OpenAI model and API key without restarting the service.
              </Typography>

              <Box mb={2}>
                <Typography variant="subtitle2" gutterBottom>
                  Hot-reloadable settings:
                </Typography>
                <ul style={{ margin: 0, paddingLeft: '20px' }}>
                  <li><code>OPENAI_MODEL</code> - Change AI model for reviews</li>
                  <li><code>OPENAI_API_KEY</code> - Update OpenAI API key</li>
                  <li><code>GITLAB_WEBHOOK_SECRET</code> - Update webhook secret</li>
                </ul>
              </Box>

              <Button
                variant="contained"
                startIcon={<RefreshIcon />}
                onClick={handleReload}
                disabled={loading}
              >
                {loading ? 'Reloading...' : 'Reload Configuration'}
              </Button>
            </CardContent>
          </Card>

          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}

          {reloadResult && (
            <Alert
              severity={reloadResult.success ? 'success' : 'error'}
              sx={{ mt: 2 }}
            >
              <Typography variant="body2">
                {reloadResult.message}
              </Typography>
              {reloadResult.changes && Object.keys(reloadResult.changes).length > 0 && (
                <Box mt={1}>
                  <Typography variant="body2" fontWeight="bold">
                    Changes applied:
                  </Typography>
                  {Object.entries(reloadResult.changes).map(([key, change]) => (
                    <Typography key={key} variant="body2" sx={{ fontFamily: 'monospace' }}>
                      • {key}: {change.old} → {change.new}
                    </Typography>
                  ))}
                </Box>
              )}
            </Alert>
          )}
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Environment Variables
              </Typography>
              <Typography variant="body2" color="textSecondary" paragraph>
                Set these environment variables to configure the service:
              </Typography>

              <Box sx={{ fontFamily: 'monospace', fontSize: '0.875rem' }}>
                <div>OPENAI_API_KEY=your-key</div>
                <div>OPENAI_MODEL=gpt-4-turbo-preview</div>
                <div>GITLAB_URL=https://gitlab.example.com</div>
                <div>GITLAB_TOKEN=your-token</div>
                <div>GITLAB_WEBHOOK_SECRET=your-secret</div>
                <div>API_AUTH_TOKEN=your-auth-token</div>
              </Box>

              <Divider sx={{ my: 2 }} />

              <Typography variant="body2" color="textSecondary">
                <strong>Note:</strong> Changes to environment variables require either
                a configuration reload (for hot-reloadable settings) or a service restart.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Settings;