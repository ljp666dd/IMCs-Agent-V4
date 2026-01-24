'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import { AgentService, Material } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Database, Atom } from "lucide-react";
import Link from 'next/link';
import StructureViewer from '@/components/structure-viewer';

export default function MaterialDetailPage() {
    const params = useParams();
    const id = params?.id as string;

    const [material, setMaterial] = useState<Material | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!id) return;

        const fetchData = async () => {
            try {
                const data = await AgentService.getMaterialDetails(id);
                setMaterial(data);
            } catch (e) {
                console.error("Failed to fetch material details", e);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [id]);

    if (loading) {
        return (
            <div className="min-h-screen bg-neutral-50 flex items-center justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            </div>
        );
    }

    if (!material) {
        return (
            <div className="min-h-screen bg-neutral-50 flex flex-col items-center justify-center text-neutral-500">
                <p>Material not found</p>
                <Link href="/materials" className="mt-4 text-blue-600 hover:underline">
                    Back to Browser
                </Link>
            </div>
        );
    }

    return (
        <main className="min-h-screen bg-neutral-50 p-8">
            <div className="max-w-7xl mx-auto space-y-6">

                {/* Header */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                        <Link href="/materials" className="p-2 hover:bg-neutral-200 rounded-full transition-colors">
                            <ArrowLeft className="h-6 w-6 text-neutral-600" />
                        </Link>
                        <div>
                            <h1 className="text-3xl font-bold text-neutral-900">{material.formula}</h1>
                            <p className="text-neutral-500 font-mono text-sm">{material.material_id}</p>
                        </div>
                    </div>

                    <Button onClick={() => window.alert('Export not implemented')} className="bg-white text-neutral-900 border border-neutral-300 hover:bg-neutral-100">
                        Export CIF
                    </Button>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Left: Properties */}
                    <Card className="lg:col-span-1">
                        <CardHeader>
                            <CardTitle>Properties</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="p-4 bg-blue-50 rounded-lg border border-blue-100">
                                <div className="text-sm text-blue-600 mb-1">Formation Energy</div>
                                <div className="text-2xl font-bold text-blue-900">
                                    {material.formation_energy?.toFixed(3) ?? "N/A"} <span className="text-sm font-normal text-blue-600">eV/atom</span>
                                </div>
                            </div>

                            <div className="space-y-2">
                                <div className="flex justify-between py-2 border-b">
                                    <span className="text-neutral-500">Formula Reduced</span>
                                    <span className="font-medium">{material.formula}</span>
                                </div>
                                <div className="flex justify-between py-2 border-b">
                                    <span className="text-neutral-500">Stability</span>
                                    <span className="font-medium text-green-600">Stable</span>
                                </div>
                                <div className="flex justify-between py-2 border-b">
                                    <span className="text-neutral-500">Symmetry</span>
                                    <span className="font-medium">N/A</span>
                                </div>
                            </div>
                        </CardContent>
                    </Card>

                    {/* Right: 3D Visualization */}
                    <Card className="lg:col-span-2 overflow-hidden flex flex-col h-[500px]">
                        <CardHeader className="bg-neutral-900 text-white flex flex-row justify-between items-center py-4">
                            <CardTitle className="text-base flex items-center gap-2">
                                <Atom className="h-4 w-4" /> Crystal Structure
                            </CardTitle>
                            <div className="text-xs text-neutral-400">Interactive 3D View</div>
                        </CardHeader>
                        <div className="flex-1 bg-neutral-100 relative">
                            {material.cif_content ? (
                                <StructureViewer
                                    cifContent={material.cif_content}
                                    className="w-full h-full border-0 rounded-none"
                                />
                            ) : (
                                <div className="absolute inset-0 flex items-center justify-center text-neutral-400">
                                    CIF Data Unavailable
                                </div>
                            )}
                        </div>
                    </Card>
                </div>

                {/* Evidence Chain (M3) */}
                <Card>
                    <CardHeader className="flex flex-row items-center gap-2">
                        <Database className="h-5 w-5 text-purple-600" />
                        <CardTitle>Evidence Chain</CardTitle>
                    </CardHeader>
                    <CardContent>
                        {!material.evidence || material.evidence.length === 0 ? (
                            <div className="text-neutral-500 text-sm p-4 bg-neutral-100 rounded text-center">
                                No linked evidence found for this material.
                            </div>
                        ) : (
                            <div className="space-y-3">
                                {material.evidence.map((ev, i) => (
                                    <div key={i} className="flex items-start gap-4 p-3 border rounded-lg bg-white hover:shadow-sm transition-shadow">
                                        <div className={`p-2 rounded-full shrink-0 ${ev.source_type === 'experiment' ? 'bg-green-100 text-green-700' :
                                                ev.source_type === 'literature' ? 'bg-amber-100 text-amber-700' :
                                                    ev.source_type === 'ml_prediction' ? 'bg-purple-100 text-purple-700' :
                                                        'bg-blue-100 text-blue-700'
                                            }`}>
                                            {ev.source_type === 'experiment' ? <Database size={16} /> : <Database size={16} />}
                                            {/* Ideally different icons */}
                                        </div>
                                        <div className="flex-1">
                                            <div className="flex justify-between">
                                                <h4 className="font-semibold text-sm capitalize">{ev.source_type}</h4>
                                                <span className="text-xs text-neutral-400">{new Date(ev.created_at).toLocaleDateString()}</span>
                                            </div>
                                            <p className="text-xs text-neutral-600 font-mono mt-1">ID: {ev.source_id}</p>

                                            {ev.metadata && (
                                                <div className="mt-2 text-xs bg-neutral-50 p-2 rounded border">
                                                    {JSON.parse(ev.metadata).filename && (
                                                        <div className="font-medium">File: {JSON.parse(ev.metadata).filename.split('\\').pop()}</div>
                                                    )}
                                                    {/* Render other metadata generically */}
                                                    <div className="opacity-75 overflow-x-auto">
                                                        {JSON.stringify(JSON.parse(ev.metadata), null, 1)}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                        <div className="flex flex-col items-end">
                                            <span className="text-xs font-bold text-neutral-400">Score</span>
                                            <span className="text-sm font-bold text-green-600">{ev.score.toFixed(1)}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>
        </main>
    );
}
