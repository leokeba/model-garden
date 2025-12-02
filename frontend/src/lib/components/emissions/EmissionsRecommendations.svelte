<script lang="ts">
    import Badge from "$lib/components/Badge.svelte";
    import Card from "$lib/components/Card.svelte";

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
        recommendations: Recommendation[];
        analyticsSummary: AnalyticsSummary | null;
        loading: boolean;
    }

    let { recommendations, analyticsSummary, loading }: Props = $props();

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

    function getEfficiencyScoreColor(score: number): string {
        if (score >= 80) return "text-green-600";
        if (score >= 60) return "text-yellow-600";
        return "text-red-600";
    }
</script>

<!-- Summary Card -->
{#if analyticsSummary}
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <Card>
            <div class="p-6 text-center">
                <div class="text-4xl mb-2">⚡</div>
                <div
                    class="text-3xl font-bold {getEfficiencyScoreColor(
                        analyticsSummary.efficiency_score,
                    )}"
                >
                    {analyticsSummary.efficiency_score}/100
                </div>
                <div class="text-sm text-gray-500">Efficiency Score</div>
            </div>
        </Card>
        <Card>
            <div class="p-6 text-center">
                <div class="text-4xl mb-2">💡</div>
                <div class="text-3xl font-bold text-gray-900">
                    {analyticsSummary.recommendation_count}
                </div>
                <div class="text-sm text-gray-500">Recommendations</div>
            </div>
        </Card>
        <Card>
            <div class="p-6 text-center">
                <div class="text-4xl mb-2">🌱</div>
                <div class="text-3xl font-bold text-green-600">
                    -{formatNumber(
                        analyticsSummary.total_potential_savings_kg,
                        3,
                    )} kg
                </div>
                <div class="text-sm text-gray-500">Potential CO₂ Savings</div>
            </div>
        </Card>
    </div>
{/if}

<!-- Recommendations List -->
<Card>
    <div class="p-6">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">
            All Recommendations
        </h3>
        {#if loading}
            <div class="flex justify-center items-center h-32">
                <div
                    class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"
                ></div>
            </div>
        {:else if recommendations.length > 0}
            <div class="space-y-4">
                {#each recommendations as rec}
                    <div
                        class="p-4 border rounded-lg {getPriorityColor(
                            rec.priority,
                        )}"
                    >
                        <div class="flex items-start gap-4">
                            <div
                                class="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center
                {rec.priority === 'high'
                                    ? 'bg-red-200'
                                    : rec.priority === 'medium'
                                      ? 'bg-yellow-200'
                                      : 'bg-blue-200'}"
                            >
                                {rec.priority === "high"
                                    ? "🔴"
                                    : rec.priority === "medium"
                                      ? "🟡"
                                      : rec.priority === "low"
                                        ? "🔵"
                                        : "ℹ️"}
                            </div>
                            <div class="flex-1">
                                <div class="flex items-center gap-2">
                                    <h4 class="font-semibold text-gray-900">
                                        {rec.title}
                                    </h4>
                                    <Badge
                                        variant={rec.priority === "high"
                                            ? "error"
                                            : rec.priority === "medium"
                                              ? "warning"
                                              : "info"}
                                    >
                                        {rec.priority}
                                    </Badge>
                                </div>
                                <p class="text-sm text-gray-600 mt-1">
                                    {rec.description}
                                </p>
                                <div class="flex items-center gap-4 mt-3">
                                    <div
                                        class="flex items-center gap-2 text-sm"
                                    >
                                        <span class="text-gray-500"
                                            >Action:</span
                                        >
                                        <span class="font-medium text-gray-700"
                                            >{rec.action}</span
                                        >
                                    </div>
                                    {#if rec.potential_savings_kg}
                                        <div
                                            class="flex items-center gap-2 text-sm"
                                        >
                                            <span class="text-gray-500"
                                                >Potential savings:</span
                                            >
                                            <span
                                                class="font-medium text-green-600"
                                                >-{formatNumber(
                                                    rec.potential_savings_kg,
                                                    4,
                                                )} kg CO₂</span
                                            >
                                        </div>
                                    {/if}
                                </div>
                            </div>
                        </div>
                    </div>
                {/each}
            </div>
        {:else}
            <div class="text-center py-12">
                <div class="text-6xl mb-4">🎉</div>
                <h3 class="text-xl font-semibold text-gray-700 mb-2">
                    You're doing great!
                </h3>
                <p class="text-gray-500">
                    No specific recommendations at this time.
                </p>
            </div>
        {/if}
    </div>
</Card>
