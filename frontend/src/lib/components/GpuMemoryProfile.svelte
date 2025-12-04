<script lang="ts">
    import {
        api,
        type GPULiveStats,
        type GPUMemoryProfile,
    } from "$lib/api/client";
    import Card from "$lib/components/Card.svelte";
    import { onDestroy, onMount } from "svelte";

    // Props
    interface Props {
        /** Auto-refresh interval in ms (0 to disable) */
        refreshInterval?: number;
        /** Show live stats section */
        showLiveStats?: boolean;
        /** Show profile breakdown section */
        showBreakdown?: boolean;
        /** Compact mode */
        compact?: boolean;
        /** Class to add to the root element */
        class?: string;
    }

    let {
        refreshInterval = 5000,
        showLiveStats = true,
        showBreakdown = true,
        compact = false,
        class: className = "",
    }: Props = $props();

    // State
    let liveStats = $state<GPULiveStats | null>(null);
    let profile = $state<GPUMemoryProfile | null>(null);
    let loading = $state(true);
    let error = $state("");
    let refreshTimer: ReturnType<typeof setInterval> | null = null;

    function formatGb(value: number): string {
        return `${value.toFixed(2)} GB`;
    }

    function formatPercent(value: number): string {
        return `${value.toFixed(1)}%`;
    }

    function getBarColor(percent: number): string {
        if (percent < 50) return "bg-green-500";
        if (percent < 75) return "bg-yellow-500";
        if (percent < 90) return "bg-orange-500";
        return "bg-red-500";
    }

    function getBreakdownItems(profile: GPUMemoryProfile) {
        const breakdown = profile.breakdown;
        const total = breakdown.total_used_gb;

        return [
            {
                label: "Model Weights",
                value: breakdown.weights_gb,
                percent: total > 0 ? (breakdown.weights_gb / total) * 100 : 0,
                color: "bg-blue-500",
                description:
                    profile.weight_file_size_gb > 0
                        ? `Files: ${formatGb(profile.weight_file_size_gb)}`
                        : undefined,
            },
            {
                label: "KV Cache",
                value: breakdown.kv_cache_gb,
                percent: total > 0 ? (breakdown.kv_cache_gb / total) * 100 : 0,
                color: "bg-purple-500",
                description:
                    profile.kv_cache.tokens > 0
                        ? `${profile.kv_cache.tokens.toLocaleString()} tokens`
                        : undefined,
            },
            {
                label: "CUDA Graphs",
                value: breakdown.cuda_graphs_gb,
                percent:
                    total > 0 ? (breakdown.cuda_graphs_gb / total) * 100 : 0,
                color: "bg-cyan-500",
                description: profile.config.enforce_eager
                    ? "(disabled)"
                    : undefined,
            },
            {
                label: "Other/Buffers",
                value: breakdown.other_gb,
                percent: total > 0 ? (breakdown.other_gb / total) * 100 : 0,
                color: "bg-gray-500",
            },
        ].filter((item) => item.value > 0.01); // Filter out very small values
    }

    async function loadData() {
        try {
            error = "";
            const stats = await api.getGpuMemoryStats();
            liveStats = stats.live;
            profile = stats.profile;
        } catch (err) {
            error =
                err instanceof Error ? err.message : "Failed to load GPU stats";
        } finally {
            loading = false;
        }
    }

    function startRefresh() {
        if (refreshInterval > 0 && !refreshTimer) {
            refreshTimer = setInterval(loadData, refreshInterval);
        }
    }

    function stopRefresh() {
        if (refreshTimer) {
            clearInterval(refreshTimer);
            refreshTimer = null;
        }
    }

    onMount(() => {
        loadData();
        startRefresh();
    });

    onDestroy(() => {
        stopRefresh();
    });
</script>

<div class={className}>
    {#if loading}
        <div class="flex items-center justify-center py-4">
            <div
                class="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600"
            ></div>
        </div>
    {:else if error && !liveStats}
        <div
            class="p-4 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700"
        >
            {error}
        </div>
    {:else}
        <!-- Live Stats -->
        {#if showLiveStats && liveStats && !liveStats.error}
            <Card class={compact ? "mb-3" : "mb-4"}>
                <div class={compact ? "p-3" : "p-4"}>
                    <div class="flex items-center justify-between mb-3">
                        <h4
                            class="text-sm font-semibold text-gray-900 flex items-center gap-2"
                        >
                            <span>📊</span>
                            <span>GPU Memory</span>
                        </h4>
                        <span class="text-xs text-gray-500">
                            {liveStats.gpu_name}
                        </span>
                    </div>

                    <!-- Memory bar -->
                    <div class="mb-3">
                        <div
                            class="flex justify-between text-xs text-gray-600 mb-1"
                        >
                            <span>{formatGb(liveStats.allocated_gb)} used</span>
                            <span>{formatGb(liveStats.free_gb)} free</span>
                        </div>
                        <div
                            class="h-3 bg-gray-200 rounded-full overflow-hidden"
                        >
                            <div
                                class={`h-full transition-all duration-300 ${getBarColor(liveStats.utilization_percent)}`}
                                style="width: {liveStats.utilization_percent}%"
                            ></div>
                        </div>
                        <div
                            class="flex justify-between text-xs text-gray-500 mt-1"
                        >
                            <span
                                >{formatPercent(
                                    liveStats.utilization_percent,
                                )}</span
                            >
                            <span>of {formatGb(liveStats.total_gb)}</span>
                        </div>
                    </div>

                    <!-- Peak stats -->
                    {#if !compact}
                        <div class="grid grid-cols-2 gap-3 text-xs">
                            <div class="bg-gray-50 rounded p-2">
                                <span class="text-gray-500">Peak Allocated</span
                                >
                                <span class="block font-semibold text-gray-700">
                                    {formatGb(liveStats.peak_allocated_gb)}
                                </span>
                            </div>
                            <div class="bg-gray-50 rounded p-2">
                                <span class="text-gray-500">Reserved</span>
                                <span class="block font-semibold text-gray-700">
                                    {formatGb(liveStats.reserved_gb)}
                                </span>
                            </div>
                        </div>
                    {/if}
                </div>
            </Card>
        {/if}

        <!-- Memory Profile Breakdown -->
        {#if showBreakdown && profile}
            <Card>
                <div class={compact ? "p-3" : "p-4"}>
                    <div class="flex items-center justify-between mb-3">
                        <h4
                            class="text-sm font-semibold text-gray-900 flex items-center gap-2"
                        >
                            <span>🧠</span>
                            <span>Memory Breakdown</span>
                        </h4>
                        {#if profile.timing.load_time_seconds > 0}
                            <span class="text-xs text-gray-500">
                                Loaded in {profile.timing.load_time_seconds.toFixed(
                                    1,
                                )}s
                            </span>
                        {/if}
                    </div>

                    <!-- Stacked bar visualization -->
                    <div class="mb-4">
                        <div
                            class="h-6 bg-gray-100 rounded-lg overflow-hidden flex"
                        >
                            {#each getBreakdownItems(profile) as item}
                                {#if item.percent > 0}
                                    <div
                                        class={`${item.color} transition-all duration-300 flex items-center justify-center`}
                                        style="width: {item.percent}%"
                                        title="{item.label}: {formatGb(
                                            item.value,
                                        )} ({formatPercent(item.percent)})"
                                    >
                                        {#if item.percent > 10}
                                            <span
                                                class="text-xs text-white font-medium truncate px-1"
                                            >
                                                {formatPercent(item.percent)}
                                            </span>
                                        {/if}
                                    </div>
                                {/if}
                            {/each}
                        </div>
                    </div>

                    <!-- Legend -->
                    <div class="space-y-2">
                        {#each getBreakdownItems(profile) as item}
                            <div
                                class="flex items-center justify-between text-sm"
                            >
                                <div class="flex items-center gap-2">
                                    <div
                                        class={`w-3 h-3 rounded ${item.color}`}
                                    ></div>
                                    <span class="text-gray-700"
                                        >{item.label}</span
                                    >
                                    {#if item.description}
                                        <span class="text-xs text-gray-400">
                                            {item.description}
                                        </span>
                                    {/if}
                                </div>
                                <span
                                    class="font-medium text-gray-900 tabular-nums"
                                >
                                    {formatGb(item.value)}
                                </span>
                            </div>
                        {/each}

                        <!-- Total -->
                        <div
                            class="flex items-center justify-between text-sm pt-2 border-t border-gray-200"
                        >
                            <span class="font-semibold text-gray-900"
                                >Total Used</span
                            >
                            <span
                                class="font-semibold text-gray-900 tabular-nums"
                            >
                                {formatGb(profile.breakdown.total_used_gb)}
                                <span class="text-gray-500 font-normal">
                                    ({formatPercent(
                                        profile.utilization.used_percent,
                                    )})
                                </span>
                            </span>
                        </div>
                        <div class="flex items-center justify-between text-sm">
                            <span class="text-gray-600">Available</span>
                            <span
                                class="text-green-600 font-medium tabular-nums"
                            >
                                {formatGb(profile.breakdown.available_gb)}
                            </span>
                        </div>
                    </div>

                    <!-- Config info -->
                    {#if !compact && profile.config.model_path}
                        <div class="mt-4 pt-3 border-t border-gray-100">
                            <div class="text-xs text-gray-500 space-y-1">
                                <div class="flex justify-between">
                                    <span>Model:</span>
                                    <span
                                        class="font-medium text-gray-700 truncate ml-2"
                                        title={profile.config.model_path}
                                    >
                                        {profile.config.model_path
                                            .split("/")
                                            .pop()}
                                    </span>
                                </div>
                                {#if profile.kv_cache.max_concurrency > 0}
                                    <div class="flex justify-between">
                                        <span>Max Concurrency:</span>
                                        <span class="font-medium text-gray-700">
                                            {profile.kv_cache.max_concurrency.toFixed(
                                                1,
                                            )}x
                                            <span
                                                class="text-gray-400 font-normal"
                                            >
                                                (for {profile.config.max_model_len.toLocaleString()}
                                                tokens)
                                            </span>
                                        </span>
                                    </div>
                                {/if}
                                {#if profile.config.tensor_parallel_size > 1}
                                    <div class="flex justify-between">
                                        <span>GPUs:</span>
                                        <span class="font-medium text-gray-700"
                                            >{profile.config
                                                .tensor_parallel_size}</span
                                        >
                                    </div>
                                {/if}
                            </div>
                        </div>
                    {/if}
                </div>
            </Card>
        {:else if !profile && !loading}
            <Card class="bg-gray-50">
                <div class={compact ? "p-3" : "p-4"}>
                    <p class="text-sm text-gray-500 text-center">
                        Load a model to see memory breakdown
                    </p>
                </div>
            </Card>
        {/if}
    {/if}
</div>
