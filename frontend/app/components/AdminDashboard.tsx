'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { 
  Loader2, 
  Activity, 
  Users, 
  FileText, 
  AlertTriangle, 
  TrendingUp, 
  Database, 
  Server, 
  Globe, 
  RefreshCw, 
  CheckCircle, 
  AlertCircle,
  BarChart3,
  Clock,
  Download,
  Upload,
  Zap,
  Shield,
  Eye,
  Settings,
  Monitor,
  HardDrive,
  Cpu,
  MemoryStick,
  Network,
  Bug,
  UserCheck,
  FileCheck,
  DollarSign,
  Calendar
} from 'lucide-react';

interface SystemMetrics {
  total_users: number;
  active_users_24h: number;
  total_resumes_processed: number;
  resumes_processed_24h: number;
  average_processing_time: number;
  success_rate: number;
  error_rate: number;
  server_uptime: string;
  database_size: string;
  storage_used: string;
  api_calls_24h: number;
  peak_concurrent_users: number;
}

interface PerformanceMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_io: number;
  response_time_avg: number;
  throughput: number;
  error_count_24h: number;
  cache_hit_rate: number;
}

interface UserActivity {
  new_signups_24h: number;
  active_sessions: number;
  top_features_used: Array<{feature: string, usage_count: number}>;
  user_retention_rate: number;
  average_session_duration: number;
}

interface SystemIssues {
  critical_errors: Array<{id: string, message: string, timestamp: string, status: string}>;
  warnings: Array<{id: string, message: string, timestamp: string}>;
  performance_alerts: Array<{id: string, metric: string, value: number, threshold: number, timestamp: string}>;
}

interface RevenueMetrics {
  daily_revenue: number;
  monthly_revenue: number;
  conversion_rate: number;
  premium_users: number;
  churn_rate: number;
}

export default function AdminDashboard() {
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics | null>(null);
  const [userActivity, setUserActivity] = useState<UserActivity | null>(null);
  const [systemIssues, setSystemIssues] = useState<SystemIssues | null>(null);
  const [revenueMetrics, setRevenueMetrics] = useState<RevenueMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [lastUpdated, setLastUpdated] = useState<string>('');
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchSystemMetrics = async () => {
    setIsLoading(true);
    setError('');

    try {
      const token = localStorage.getItem('token');
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5000';
      
      // Fetch all metrics in parallel
      const [systemRes, performanceRes, userRes, issuesRes, revenueRes] = await Promise.all([
        fetch(`${apiUrl}/api/admin/system-metrics`, {
          headers: { 'Authorization': `Bearer ${token}` }
        }),
        fetch(`${apiUrl}/api/admin/performance-metrics`, {
          headers: { 'Authorization': `Bearer ${token}` }
        }),
        fetch(`${apiUrl}/api/admin/user-activity`, {
          headers: { 'Authorization': `Bearer ${token}` }
        }),
        fetch(`${apiUrl}/api/admin/system-issues`, {
          headers: { 'Authorization': `Bearer ${token}` }
        }),
        fetch(`${apiUrl}/api/admin/revenue-metrics`, {
          headers: { 'Authorization': `Bearer ${token}` }
        })
      ]);

      if (systemRes.ok) setSystemMetrics(await systemRes.json());
      if (performanceRes.ok) setPerformanceMetrics(await performanceRes.json());
      if (userRes.ok) setUserActivity(await userRes.json());
      if (issuesRes.ok) setSystemIssues(await issuesRes.json());
      if (revenueRes.ok) setRevenueMetrics(await revenueRes.json());

      setLastUpdated(new Date().toLocaleTimeString());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load metrics');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchSystemMetrics();
    
    // Auto-refresh every 30 seconds if enabled
    const interval = autoRefresh ? setInterval(fetchSystemMetrics, 30000) : null;
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  const getStatusColor = (value: number, thresholds: {good: number, warning: number}) => {
    if (value >= thresholds.good) return 'text-green-600';
    if (value >= thresholds.warning) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getStatusBadge = (value: number, thresholds: {good: number, warning: number}) => {
    if (value >= thresholds.good) return 'default';
    if (value >= thresholds.warning) return 'secondary';
    return 'destructive';
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  const formatBytes = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${Math.round(bytes / Math.pow(1024, i) * 100) / 100} ${sizes[i]}`;
  };

  return (
    <div className="w-full space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">Admin Dashboard</h1>
          <p className="text-gray-400">System monitoring and analytics for PassResume</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-sm text-gray-400">
            Last updated: {lastUpdated}
          </div>
          <Button
            onClick={() => setAutoRefresh(!autoRefresh)}
            variant={autoRefresh ? "default" : "outline"}
            size="sm"
          >
            <Activity className="mr-2 h-4 w-4" />
            Auto-refresh {autoRefresh ? 'ON' : 'OFF'}
          </Button>
          <Button 
            onClick={fetchSystemMetrics} 
            disabled={isLoading}
            size="sm"
          >
            {isLoading ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="mr-2 h-4 w-4" />
            )}
            Refresh
          </Button>
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Quick Stats Overview */}
      {systemMetrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="bg-gradient-to-br from-blue-500/10 to-blue-600/10 border-blue-500/20">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-blue-300">Total Users</p>
                  <p className="text-2xl font-bold text-white">{formatNumber(systemMetrics.total_users)}</p>
                  <p className="text-xs text-blue-400">+{systemMetrics.active_users_24h} active today</p>
                </div>
                <Users className="h-8 w-8 text-blue-400" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-green-500/10 to-green-600/10 border-green-500/20">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-green-300">Resumes Processed</p>
                  <p className="text-2xl font-bold text-white">{formatNumber(systemMetrics.total_resumes_processed)}</p>
                  <p className="text-xs text-green-400">+{systemMetrics.resumes_processed_24h} today</p>
                </div>
                <FileText className="h-8 w-8 text-green-400" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-purple-500/10 to-purple-600/10 border-purple-500/20">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-purple-300">Success Rate</p>
                  <p className="text-2xl font-bold text-white">{systemMetrics.success_rate}%</p>
                  <p className="text-xs text-purple-400">Error rate: {systemMetrics.error_rate}%</p>
                </div>
                <CheckCircle className="h-8 w-8 text-purple-400" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-orange-500/10 to-orange-600/10 border-orange-500/20">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-orange-300">Avg Processing</p>
                  <p className="text-2xl font-bold text-white">{systemMetrics.average_processing_time}s</p>
                  <p className="text-xs text-orange-400">{formatNumber(systemMetrics.api_calls_24h)} API calls today</p>
                </div>
                <Clock className="h-8 w-8 text-orange-400" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Main Dashboard Tabs */}
      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-6 bg-gray-800/50">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="users">Users</TabsTrigger>
          <TabsTrigger value="issues">Issues</TabsTrigger>
          <TabsTrigger value="revenue">Revenue</TabsTrigger>
          <TabsTrigger value="logs">Logs</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* System Health */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Monitor className="h-5 w-5" />
                  System Health
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {performanceMetrics && (
                  <>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">CPU Usage</span>
                      <Badge variant={getStatusBadge(100 - performanceMetrics.cpu_usage, {good: 70, warning: 50})}>
                        {performanceMetrics.cpu_usage}%
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Memory Usage</span>
                      <Badge variant={getStatusBadge(100 - performanceMetrics.memory_usage, {good: 70, warning: 50})}>
                        {performanceMetrics.memory_usage}%
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Disk Usage</span>
                      <Badge variant={getStatusBadge(100 - performanceMetrics.disk_usage, {good: 80, warning: 60})}>
                        {performanceMetrics.disk_usage}%
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Response Time</span>
                      <Badge variant={getStatusBadge(1000 - performanceMetrics.response_time_avg, {good: 800, warning: 500})}>
                        {performanceMetrics.response_time_avg}ms
                      </Badge>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>

            {/* Recent Activity */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5" />
                  Recent Activity
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {userActivity && (
                  <>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">New Signups (24h)</span>
                      <span className="font-semibold text-green-400">+{userActivity.new_signups_24h}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Active Sessions</span>
                      <span className="font-semibold text-blue-400">{userActivity.active_sessions}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Avg Session Duration</span>
                      <span className="font-semibold">{userActivity.average_session_duration}min</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">User Retention</span>
                      <Badge variant="default">{userActivity.user_retention_rate}%</Badge>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Critical Issues Alert */}
          {systemIssues && systemIssues.critical_errors.length > 0 && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                {systemIssues.critical_errors.length} critical error(s) require immediate attention
              </AlertDescription>
            </Alert>
          )}
        </TabsContent>

        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {performanceMetrics && (
              <>
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-sm">
                      <Cpu className="h-4 w-4" />
                      CPU Performance
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold mb-2">{performanceMetrics.cpu_usage}%</div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-blue-500 h-2 rounded-full" 
                        style={{width: `${performanceMetrics.cpu_usage}%`}}
                      ></div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-sm">
                      <MemoryStick className="h-4 w-4" />
                      Memory Usage
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold mb-2">{performanceMetrics.memory_usage}%</div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-green-500 h-2 rounded-full" 
                        style={{width: `${performanceMetrics.memory_usage}%`}}
                      ></div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-sm">
                      <HardDrive className="h-4 w-4" />
                      Disk Usage
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold mb-2">{performanceMetrics.disk_usage}%</div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-purple-500 h-2 rounded-full" 
                        style={{width: `${performanceMetrics.disk_usage}%`}}
                      ></div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-sm">
                      <Network className="h-4 w-4" />
                      Network I/O
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold mb-2">{formatBytes(performanceMetrics.network_io)}/s</div>
                    <p className="text-sm text-gray-400">Current throughput</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-sm">
                      <Clock className="h-4 w-4" />
                      Response Time
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold mb-2">{performanceMetrics.response_time_avg}ms</div>
                    <p className="text-sm text-gray-400">Average response time</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-sm">
                      <Zap className="h-4 w-4" />
                      Cache Hit Rate
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold mb-2">{performanceMetrics.cache_hit_rate}%</div>
                    <p className="text-sm text-gray-400">Cache efficiency</p>
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        </TabsContent>

        {/* Users Tab */}
        <TabsContent value="users" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <UserCheck className="h-5 w-5" />
                  User Statistics
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {userActivity && (
                  <>
                    <div className="flex justify-between">
                      <span>New Signups (24h)</span>
                      <span className="font-bold text-green-400">+{userActivity.new_signups_24h}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Active Sessions</span>
                      <span className="font-bold">{userActivity.active_sessions}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Retention Rate</span>
                      <Badge variant="default">{userActivity.user_retention_rate}%</Badge>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Feature Usage
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {userActivity?.top_features_used.map((feature, index) => (
                  <div key={index} className="flex justify-between items-center">
                    <span className="text-sm">{feature.feature}</span>
                    <Badge variant="outline">{formatNumber(feature.usage_count)}</Badge>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Issues Tab */}
        <TabsContent value="issues" className="space-y-6">
          {systemIssues && (
            <>
              {/* Critical Errors */}
              {systemIssues.critical_errors.length > 0 && (
                <Card className="border-red-500/50">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-red-400">
                      <AlertTriangle className="h-5 w-5" />
                      Critical Errors ({systemIssues.critical_errors.length})
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {systemIssues.critical_errors.map((error) => (
                      <div key={error.id} className="bg-red-500/10 border border-red-500/20 rounded-lg p-3">
                        <div className="flex justify-between items-start">
                          <div>
                            <p className="font-medium text-red-300">{error.message}</p>
                            <p className="text-xs text-gray-400">{new Date(error.timestamp).toLocaleString()}</p>
                          </div>
                          <Badge variant="destructive">{error.status}</Badge>
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              )}

              {/* Warnings */}
              {systemIssues.warnings.length > 0 && (
                <Card className="border-yellow-500/50">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-yellow-400">
                      <AlertCircle className="h-5 w-5" />
                      Warnings ({systemIssues.warnings.length})
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {systemIssues.warnings.map((warning) => (
                      <div key={warning.id} className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-3">
                        <p className="font-medium text-yellow-300">{warning.message}</p>
                        <p className="text-xs text-gray-400">{new Date(warning.timestamp).toLocaleString()}</p>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              )}

              {/* Performance Alerts */}
              {systemIssues.performance_alerts.length > 0 && (
                <Card className="border-orange-500/50">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-orange-400">
                      <TrendingUp className="h-5 w-5" />
                      Performance Alerts ({systemIssues.performance_alerts.length})
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {systemIssues.performance_alerts.map((alert) => (
                      <div key={alert.id} className="bg-orange-500/10 border border-orange-500/20 rounded-lg p-3">
                        <div className="flex justify-between items-start">
                          <div>
                            <p className="font-medium text-orange-300">
                              {alert.metric}: {alert.value} (threshold: {alert.threshold})
                            </p>
                            <p className="text-xs text-gray-400">{new Date(alert.timestamp).toLocaleString()}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </TabsContent>

        {/* Revenue Tab */}
        <TabsContent value="revenue" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {revenueMetrics && (
              <>
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-sm">
                      <DollarSign className="h-4 w-4" />
                      Daily Revenue
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">${revenueMetrics.daily_revenue.toLocaleString()}</div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-sm">
                      <Calendar className="h-4 w-4" />
                      Monthly Revenue
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">${revenueMetrics.monthly_revenue.toLocaleString()}</div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-sm">
                      <TrendingUp className="h-4 w-4" />
                      Conversion Rate
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{revenueMetrics.conversion_rate}%</div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-sm">
                      <Shield className="h-4 w-4" />
                      Premium Users
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{formatNumber(revenueMetrics.premium_users)}</div>
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        </TabsContent>

        {/* Logs Tab */}
        <TabsContent value="logs" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Eye className="h-5 w-5" />
                System Logs
              </CardTitle>
              <CardDescription>
                Real-time system logs and application events
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="bg-gray-900 rounded-lg p-4 font-mono text-sm max-h-96 overflow-y-auto">
                <div className="text-green-400">[INFO] System monitoring dashboard loaded</div>
                <div className="text-blue-400">[DEBUG] Fetching system metrics...</div>
                <div className="text-yellow-400">[WARN] High memory usage detected: 85%</div>
                <div className="text-green-400">[INFO] User authentication successful</div>
                <div className="text-red-400">[ERROR] Database connection timeout</div>
                <div className="text-green-400">[INFO] Resume processing completed successfully</div>
                <div className="text-blue-400">[DEBUG] Cache hit rate: 92%</div>
                <div className="text-green-400">[INFO] PDF export completed</div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}