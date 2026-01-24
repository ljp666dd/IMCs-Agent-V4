'use client';

import { useEffect, useState } from 'react';
import { AgentService, Material } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ArrowLeft, Loader2, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import Link from 'next/link';

export default function MaterialsPage() {
    const [materials, setMaterials] = useState<Material[]>([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState("");

    const filteredMaterials = materials.filter(m =>
        m.formula.toLowerCase().includes(search.toLowerCase()) ||
        m.material_id.toLowerCase().includes(search.toLowerCase())
    );

    useEffect(() => {
        const fetchMaterials = async () => {
            try {
                const data = await AgentService.getMaterials();
                setMaterials(data);
            } catch (e) {
                console.error("Failed to fetch materials", e);
            } finally {
                setLoading(false);
            }
        };
        fetchMaterials();
    }, []);

    return (
        <main className="min-h-screen bg-neutral-50 p-8">
            <div className="max-w-7xl mx-auto space-y-6">

                {/* Header */}
                <div className="flex items-center space-x-4">
                    <Link href="/" className="p-2 hover:bg-neutral-200 rounded-full transition-colors">
                        <ArrowLeft className="h-6 w-6 text-neutral-600" />
                    </Link>
                    <h1 className="text-2xl font-bold text-neutral-900">Materials Database</h1>
                </div>

                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle>Stored Structures ({filteredMaterials.length})</CardTitle>
                        <div className="relative w-64">
                            <Search className="absolute left-2 top-2.5 h-4 w-4 text-neutral-500" />
                            <Input
                                placeholder="Search formula..."
                                className="pl-8"
                                value={search}
                                onChange={(e) => setSearch(e.target.value)}
                            />
                        </div>
                    </CardHeader>
                    <CardContent>
                        {loading ? (
                            <div className="flex justify-center p-8">
                                <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
                            </div>
                        ) : filteredMaterials.length === 0 ? (
                            <div className="text-center p-8 text-neutral-500">
                                No materials found. Try running a theory task first.
                            </div>
                        ) : (
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm text-left">
                                    <thead className="text-xs text-neutral-500 uppercase bg-neutral-100">
                                        <tr>
                                            <th className="px-6 py-3">ID</th>
                                            <th className="px-6 py-3">Formula</th>
                                            <th className="px-6 py-3">Energy (eV/atom)</th>
                                            <th className="px-6 py-3">CIF Path</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {filteredMaterials.map((mat) => (
                                            <tr key={mat.id} className="bg-white border-b hover:bg-neutral-50 transition-colors">
                                                <td className="px-6 py-4 font-medium text-neutral-900">
                                                    <Link href={`/materials/${mat.material_id}`} className="text-blue-600 hover:underline">
                                                        {mat.material_id}
                                                    </Link>
                                                </td>
                                                <td className="px-6 py-4">
                                                    {mat.formula}
                                                </td>
                                                <td className="px-6 py-4">
                                                    {mat.formation_energy?.toFixed(3) ?? "N/A"}
                                                </td>
                                                <td className="px-6 py-4 text-neutral-400 font-mono text-xs truncate max-w-[200px]">
                                                    {mat.cif_path}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>
        </main>
    );
}
