'use client';

import { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { AgentService } from "@/lib/api";
import Link from 'next/link';
import { Activity, Database, Server, Cpu, MessageSquare, Beaker, Brain } from "lucide-react";

export default function Home() {
  const [status, setStatus] = useState<any>(null);
  const [theoryStatus, setTheoryStatus] = useState<any>(null);

  useEffect(() => {
    // Poll status
    const fetchData = async () => {
      try {
        const s = await AgentService.getSystemStatus();
        setStatus(s);
        const t = await AgentService.getTheoryStatus();
        setTheoryStatus(t);
      } catch (e) {
        console.error("Failed to fetch status", e);
      }
    };
    fetchData();
  }, []);

  return (
    <main className="min-h-screen bg-neutral-50 p-8">
      <div className="max-w-7xl mx-auto space-y-8">

        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-neutral-900 tracking-tight">IMCs Engineering Platform</h1>
            <p className="text-neutral-500 mt-1">Intelligent Materials Catalyst System v3.3</p>
          </div>
          <div className="flex items-center space-x-2 text-sm text-green-600 bg-green-50 px-3 py-1 rounded-full border border-green-200">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
            </span>
            <span className="font-medium">System Operational</span>
          </div>
        </div>

        {/* Status Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">API Status</CardTitle>
              <Server className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{status ? "Online" : "Connecting..."}</div>
              <p className="text-xs text-muted-foreground">Backend v{status?.version || '...'}</p>
            </CardContent>
          </Card>

          <Link href="/materials" className="block transition-transform hover:scale-105">
            <Card className="cursor-pointer border-blue-200 hover:bg-blue-50">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-blue-900">Materials DB</CardTitle>
                <Database className="h-4 w-4 text-blue-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-blue-700">{theoryStatus?.cif_files || 0}</div>
                <p className="text-xs text-blue-400">View Cached Structures &rarr;</p>
              </CardContent>
            </Card>
          </Link>

          <Link href="/chat" className="block transition-transform hover:scale-105">
            <Card className="cursor-pointer border-green-200 hover:bg-green-50">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-green-900">Research Agent</CardTitle>
                <MessageSquare className="h-4 w-4 text-green-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-700">Chat</div>
                <p className="text-xs text-green-400">Ask for plans & analysis &rarr;</p>
              </CardContent>
            </Card>
          </Link>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Training Jobs</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">0</div>
              <p className="text-xs text-muted-foreground">Active processes</p>
            </CardContent>
          </Card>
        </div>

        {/* Recent Activity / Graph Placeholder */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Link href="/experiments" className="block transition-transform hover:scale-105">
            <Card className="cursor-pointer border-purple-200 hover:bg-purple-50">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-purple-900">Experiments</CardTitle>
                <Beaker className="h-4 w-4 text-purple-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-purple-700">Workbench</div>
                <p className="text-xs text-purple-400">Upload Data & Analyze &rarr;</p>
              </CardContent>
            </Card>
          </Link>

          <Link href="/ml" className="block transition-transform hover:scale-105">
            <Card className="cursor-pointer border-blue-200 hover:bg-blue-50">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-blue-900">Machine Learning</CardTitle>
                <Brain className="h-4 w-4 text-blue-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-blue-700">Model Center</div>
                <p className="text-xs text-blue-400">Train & Explain Models &rarr;</p>
              </CardContent>
            </Card>
          </Link>

          <Card className="col-span-1">
            <CardHeader>
              <CardTitle>Model Performance</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[200px] flex items-center justify-center border-2 border-dashed rounded-md bg-neutral-50 text-neutral-400">
                No models trained
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </main>
  );
}
