'use client';

import { useState } from 'react';
import { AgentService, ExperimentResponse } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ArrowLeft, Upload, FileText, CheckCheck, AlertCircle } from "lucide-react";
import Link from 'next/link';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function ExperimentPage() {
    const [file, setFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<ExperimentResponse | null>(null);
    const [plotData, setPlotData] = useState<any[]>([]);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    };

    const handleUpload = async () => {
        if (!file) return;
        setLoading(true);
        setResult(null);
        setPlotData([]);

        try {
            const res = await AgentService.uploadExperiment(file, "lsv");
            setResult(res);

            setResult(res);

            // Parse result for plotting
            if (res.analysis?.data) {
                const { voltage, current } = res.analysis.data;
                // Basic downsampling if too large
                const step = Math.ceil(voltage.length / 500);
                const data = [];
                for (let i = 0; i < voltage.length; i += step) {
                    data.push({ v: voltage[i], i: current[i] });
                }
                setPlotData(data);
            }
        } catch (e) {
            alert("Upload failed");
        } finally {
            setLoading(false);
        }
    };

    return (
        <main className="min-h-screen bg-neutral-50 p-8">
            <div className="max-w-7xl mx-auto space-y-6">

                {/* Header */}
                <div className="flex items-center space-x-4">
                    <Link href="/" className="p-2 hover:bg-neutral-200 rounded-full transition-colors">
                        <ArrowLeft className="h-6 w-6 text-neutral-600" />
                    </Link>
                    <h1 className="text-2xl font-bold text-neutral-900">Experiment Workbench</h1>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

                    {/* Upload Card */}
                    <Card>
                        <CardHeader>
                            <CardTitle>Upload Data</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="border-2 border-dashed border-neutral-300 rounded-lg p-8 flex flex-col items-center justify-center text-center hover:bg-neutral-50 transition-colors">
                                <Upload className="h-10 w-10 text-neutral-400 mb-2" />
                                <p className="text-sm text-neutral-500 mb-4">Drag and drop CSV files here</p>
                                <Input
                                    type="file"
                                    accept=".csv,.txt"
                                    onChange={handleFileChange}
                                    className="max-w-xs"
                                />
                            </div>

                            {file && (
                                <div className="flex items-center gap-2 text-sm bg-blue-50 p-2 rounded text-blue-700">
                                    <FileText className="h-4 w-4" />
                                    <span>{file.name} ({(file.size / 1024).toFixed(1)} KB)</span>
                                </div>
                            )}

                            <Button
                                onClick={handleUpload}
                                className="w-full"
                                disabled={!file || loading}
                            >
                                {loading ? "Analyzing..." : "Analyze LSV Data"}
                            </Button>
                        </CardContent>
                    </Card>

                    {/* Results Card */}
                    <Card className="col-span-1 md:col-span-2">
                        <CardHeader>
                            <CardTitle>Analysis Results</CardTitle>
                        </CardHeader>
                        <CardContent>
                            {!result ? (
                                <div className="h-[200px] flex items-center justify-center text-neutral-400 italic">
                                    Upload a file to see results
                                </div>
                            ) : (
                                <div className="space-y-4">
                                    <div className="flex items-center gap-2 text-green-600 bg-green-50 p-3 rounded-lg border border-green-200">
                                        <CheckCheck className="h-5 w-5" />
                                        <span className="font-medium">Analysis Complete</span>
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="p-3 bg-white border rounded shadow-sm">
                                            <div className="text-xs text-neutral-500">Overpotential @ 10mA</div>
                                            <div className="text-xl font-bold text-blue-700">
                                                {result.analysis.overpotential_10mA ? `${result.analysis.overpotential_10mA.toFixed(1)} mV` : "N/A"}
                                            </div>
                                        </div>
                                        <div className="p-3 bg-white border rounded shadow-sm">
                                            <div className="text-xs text-neutral-500">Onset Potential</div>
                                            <div className="text-xl font-bold text-purple-700">
                                                {result.analysis.onset_potential ? `${result.analysis.onset_potential.toFixed(3)} V` : "N/A"}
                                            </div>
                                        </div>
                                        <div className="p-3 bg-white border rounded shadow-sm">
                                            <div className="text-xs text-neutral-500">Max Current</div>
                                            <div className="text-xl font-bold text-neutral-700">
                                                {result.analysis.current_density_max ? `${result.analysis.current_density_max.toFixed(1)} mA` : "N/A"}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="pt-4 border-t">
                                        <p className="text-sm font-medium mb-3">LSV Curve</p>
                                        <div className="h-[300px] w-full">
                                            <ResponsiveContainer width="100%" height="100%">
                                                <LineChart data={plotData} margin={{ top: 5, right: 20, bottom: 25, left: 10 }}>
                                                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                                    <XAxis
                                                        dataKey="v"
                                                        type="number"
                                                        domain={['auto', 'auto']}
                                                        tickFormatter={(val) => val.toFixed(1)}
                                                        label={{ value: 'Potential (V)', position: 'insideBottom', offset: -15 }}
                                                    />
                                                    <YAxis
                                                        label={{ value: 'J (mA/cm²)', angle: -90, position: 'insideLeft', offset: 10 }}
                                                    />
                                                    <Tooltip
                                                        formatter={(val: any) => Number(val).toFixed(2)}
                                                        labelFormatter={(val: any) => `V: ${Number(val).toFixed(2)}`}
                                                    />
                                                    <Line
                                                        type="monotone"
                                                        dataKey="i"
                                                        stroke="#2563eb"
                                                        dot={false}
                                                        strokeWidth={2}
                                                        isAnimationActive={false}
                                                    />
                                                </LineChart>
                                            </ResponsiveContainer>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </CardContent>
                    </Card>
                </div>
            </div>
        </main>
    );
}
