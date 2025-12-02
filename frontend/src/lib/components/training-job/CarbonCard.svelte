<script lang="ts">
    import Card from "$lib/components/Card.svelte";

    interface Props {
        carbonData: any;
        loading: boolean;
    }

    let { carbonData, loading }: Props = $props();
</script>

<Card>
    <div class="p-6">
        <h3 class="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <span class="mr-2">🌱</span>
            Carbon Footprint
        </h3>

        {#if loading}
            <div class="flex items-center justify-center py-4">
                <div
                    class="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600"
                ></div>
            </div>
        {:else if carbonData}
            <div class="space-y-3 text-sm">
                <div class="bg-green-50 border border-green-200 rounded-lg p-3">
                    <div class="text-xs text-green-700 font-medium mb-1">
                        Total Emissions
                    </div>
                    <div class="text-2xl font-bold text-green-900">
                        {carbonData.emissions_kg_co2
                            ? carbonData.emissions_kg_co2.toFixed(4)
                            : carbonData.emissions_kg?.toFixed(4) || "N/A"} kg
                    </div>
                    <div class="text-xs text-green-600 mt-0.5">
                        CO₂ equivalent
                    </div>
                </div>

                <div class="grid grid-cols-2 gap-3">
                    <div class="bg-gray-50 rounded-lg p-2">
                        <div class="text-xs text-gray-600">Energy Used</div>
                        <div class="text-sm font-semibold text-gray-900">
                            {carbonData.energy_consumed_kwh
                                ? carbonData.energy_consumed_kwh.toFixed(3)
                                : carbonData.energy_consumed?.toFixed(3) ||
                                  "N/A"} kWh
                        </div>
                    </div>

                    <div class="bg-gray-50 rounded-lg p-2">
                        <div class="text-xs text-gray-600">Duration</div>
                        <div class="text-sm font-semibold text-gray-900">
                            {carbonData.duration_seconds
                                ? carbonData.duration_seconds < 60
                                    ? `${Math.round(carbonData.duration_seconds)}s`
                                    : carbonData.duration_seconds < 3600
                                      ? `${Math.round(carbonData.duration_seconds / 60)}m`
                                      : `${Math.round(carbonData.duration_seconds / 3600)}h`
                                : carbonData.duration
                                  ? carbonData.duration < 60
                                      ? `${Math.round(carbonData.duration)}s`
                                      : carbonData.duration < 3600
                                        ? `${Math.round(carbonData.duration / 60)}m`
                                        : `${Math.round(carbonData.duration / 3600)}h`
                                  : "N/A"}
                        </div>
                    </div>
                </div>

                {#if carbonData.cpu_energy_kwh || carbonData.gpu_energy_kwh || carbonData.ram_energy_kwh || carbonData.cpu_energy || carbonData.gpu_energy || carbonData.ram_energy}
                    <div class="border-t pt-3">
                        <div class="text-xs text-gray-600 font-medium mb-2">
                            Energy Breakdown
                        </div>
                        <div class="space-y-1">
                            {#if carbonData.cpu_energy_kwh || carbonData.cpu_energy}
                                <div class="flex justify-between text-xs">
                                    <span class="text-gray-600">CPU:</span>
                                    <span class="font-medium"
                                        >{(
                                            carbonData.cpu_energy_kwh ||
                                            carbonData.cpu_energy ||
                                            0
                                        ).toFixed(3)} kWh</span
                                    >
                                </div>
                            {/if}
                            {#if carbonData.gpu_energy_kwh || carbonData.gpu_energy}
                                <div class="flex justify-between text-xs">
                                    <span class="text-gray-600">GPU:</span>
                                    <span class="font-medium"
                                        >{(
                                            carbonData.gpu_energy_kwh ||
                                            carbonData.gpu_energy ||
                                            0
                                        ).toFixed(3)} kWh</span
                                    >
                                </div>
                            {/if}
                            {#if carbonData.ram_energy_kwh || carbonData.ram_energy}
                                <div class="flex justify-between text-xs">
                                    <span class="text-gray-600">RAM:</span>
                                    <span class="font-medium"
                                        >{(
                                            carbonData.ram_energy_kwh ||
                                            carbonData.ram_energy ||
                                            0
                                        ).toFixed(3)} kWh</span
                                    >
                                </div>
                            {/if}
                        </div>
                    </div>
                {/if}

                {#if carbonData.equivalents}
                    <div class="border-t pt-3">
                        <div class="text-xs text-gray-600 font-medium mb-2">
                            Equivalents
                        </div>
                        <div class="space-y-1 text-xs text-gray-700">
                            {#if carbonData.equivalents.km_driven}
                                <div>
                                    🚗 {carbonData.equivalents.km_driven.toFixed(
                                        2,
                                    )} km driven
                                </div>
                            {/if}
                            {#if carbonData.equivalents.smartphones_charged}
                                <div>
                                    📱 {Math.round(
                                        carbonData.equivalents
                                            .smartphones_charged,
                                    )} smartphones charged
                                </div>
                            {/if}
                            {#if carbonData.equivalents.tree_months}
                                <div>
                                    🌳 {carbonData.equivalents.tree_months.toFixed(
                                        1,
                                    )} tree-months to offset
                                </div>
                            {/if}
                        </div>
                    </div>
                {/if}
            </div>
        {:else}
            <p class="text-sm text-gray-500 text-center py-4">
                No carbon data available for this job yet.
            </p>
        {/if}
    </div>
</Card>
