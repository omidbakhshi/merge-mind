import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  CircularProgress,
  Alert,
} from '@mui/material';
import { Assessment as AssessmentIcon } from '@mui/icons-material';
import { apiService } from '../services/api';

const Reviews: React.FC = () => {
  const [activeReviews, setActiveReviews] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadReviews = async () => {
      try {
        setLoading(true);
        const data = await apiService.getActiveReviews();
        setActiveReviews(data.active_reviews || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load reviews');
      } finally {
        setLoading(false);
      }
    };

    loadReviews();
  }, []);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Active Reviews
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {activeReviews.map((review, index) => (
          <Grid item xs={12} md={6} lg={4} key={index}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <AssessmentIcon sx={{ mr: 1, color: 'primary.main' }} />
                  <Typography variant="h6" component="h2">
                    MR #{review.mr_iid}
                  </Typography>
                </Box>

                <Typography color="textSecondary" gutterBottom>
                  Project: {review.project_id}
                </Typography>

                <Chip
                  label={review.status.toUpperCase()}
                  color={review.status === 'completed' ? 'success' : 'warning'}
                  size="small"
                />
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {activeReviews.length === 0 && !loading && (
        <Box textAlign="center" py={8}>
          <AssessmentIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="textSecondary">
            No active reviews
          </Typography>
          <Typography variant="body2" color="textSecondary">
            All reviews have been completed
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default Reviews;