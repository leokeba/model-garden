<script lang="ts">
    import Card from "$lib/components/Card.svelte";

    interface Insight {
        type: string;
        title: string;
        value: string;
        context: string;
    }

    interface Recommendation {
        id: string;
        priority: "high" | "medium" | "low" | "info";
        title: string;
        description: string;
        potential_savings_kg: number | null;
        action: string;
    }

    interface AnalyticsSummary {
        total_potential_savings_kg: number;
        efficiency_score: number;
        recommendation_count: number;
    }

    interface Props {
        insights: Insight[];
        recommendations: Recommendation[];
        analyticsSummary: AnalyticsSummary | null;
        totalEmissions: number;
        onViewAllRecommendations: () => void;
    }

    let {
        insights,
        recommendations,
        analyticsSummary,
        totalEmissions,
        onViewAllRecommendations,
    }: Props = $props();

    function formatNumber(num: number, decimals = 2): string {
        return num.toFixed(decimals);
    }

    function getPriorityColor(priority: string): string {
        switch (priority) {
            case "high":
                return "bg-red-100 text-red-800 border-red-200";
            case "medium":
                return "bg-yellow-100 text-yellow-800 border-yellow-200";
            case "low":
                return "bg-blue-100 text-blue-800 border-blue-200";
            default:
                return "bg-gray-100 text-gray-800 border-gray-200";
        }
    }
</script>

<!-- Quick Insights & Top Recommendations -->
<div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
    <!-- Insights -->
    <Card>
        <div class="p-6">
            <h3 class="text-lg font-semibold text-gray-900 mb-4">
                📊 Key Insights
            </h3>
            {#if insights.length > 0}
                <div class="space-y-4">
                    {#each insights as insight}
                        <div class="p-4 bg-gray-50 rounded-lg">
                            <div class="text-sm text-gray-600">
                                {insight.title}
                            </div>
                            <div class="text-2xl font-bold text-gray-900">
                                {insight.value}
                            </div>
                            <div class="text-xs text-gray-500 mt-1">
                                {insight.context}
                            </div>
                        </div>
                    {/each}
                </div>
            {:else}
                <p class="text-gray-500 text-center py-8">
                    Run some jobs to see insights
                </p>
            {/if}
        </div>
    </Card>

    <!-- Top Recommendations -->
    <Card>
        <div class="p-6">
            <div class="flex justify-between items-center mb-4">
                <h3 class="text-lg font-semibold text-gray-900">
                    💡 Top Recommendations
                </h3>
                {#if analyticsSummary && analyticsSummary.total_potential_savings_kg > 0}
                    <span class="text-sm text-green-600 font-medium">
                        Potential: -{formatNumber(
                            analyticsSummary.total_potential_savings_kg,
                            3,
                        )} kg CO₂
                    </span>
                {/if}
            </div>
            {#if recommendations.length > 0}
                <div class="space-y-3">
                    {#each recommendations.slice(0, 3) as rec}
                        <div
                            class="p-4 border rounded-lg {getPriorityColor(
                                rec.priority,
                            )}"
                        >
                            <div class="flex items-start gap-3">
                                <span class="text-xl">
                                    {rec.priority === "high"
                                        ? "🔴"
                                        : rec.priority === "medium"
                                          ? "🟡"
                                          : "🔵"}
                                </span>
                                <div class="flex-1">
                                    <div class="font-medium">{rec.title}</div>
                                    <div class="text-sm opacity-80 mt-1">
                                        {rec.description}
                                    </div>
                                </div>
                            </div>
                        </div>
                    {/each}
                </div>
                {#if recommendations.length > 3}
                    <button
                        onclick={onViewAllRecommendations}
                        class="mt-4 text-sm text-primary-600 hover:text-primary-800 font-medium"
                    >
                        View all {recommendations.length} recommendations →
                    </button>
                {/if}
            {:else}
                <p class="text-gray-500 text-center py-8">
                    No recommendations yet
                </p>
            {/if}
        </div>
    </Card>
</div>

<!-- Environmental Equivalents -->
<Card class="mb-8">
    <div class="p-6">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">
            🌍 Environmental Impact
        </h3>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div class="text-center p-4 bg-green-50 rounded-lg">
                <div class="text-4xl mb-2">🚗</div>
                <div class="text-2xl font-bold text-green-700">
                    {formatNumber(totalEmissions * 4.6, 1)} km
                </div>
                <div class="text-sm text-green-600">Car equivalent</div>
            </div>
            <div class="text-center p-4 bg-blue-50 rounded-lg">
                <div class="text-4xl mb-2">📱</div>
                <div class="text-2xl font-bold text-blue-700">
                    {Math.round(totalEmissions * 121)}
                </div>
                <div class="text-sm text-blue-600">Smartphones charged</div>
            </div>
            <div class="text-center p-4 bg-emerald-50 rounded-lg">
                <div class="text-4xl mb-2">🌳</div>
                <div class="text-2xl font-bold text-emerald-700">
                    {formatNumber(totalEmissions / 0.006, 1)}
                </div>
                <div class="text-sm text-emerald-600">
                    Tree-months to offset
                </div>
            </div>
        </div>
    </div>
</Card>
