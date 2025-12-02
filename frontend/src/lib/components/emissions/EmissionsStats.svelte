<script lang="ts">
    import Card from "$lib/components/Card.svelte";

    interface Props {
        totalJobs: number;
        totalEmissions: number;
        totalEnergy: number;
        avgCarbonIntensity: number;
    }

    let { totalJobs, totalEmissions, totalEnergy, avgCarbonIntensity }: Props =
        $props();

    function formatNumber(num: number, decimals = 2): string {
        return num.toFixed(decimals);
    }

    function getEmissionsColor(kg: number): string {
        if (kg < 0.01) return "text-green-600";
        if (kg < 0.1) return "text-yellow-600";
        return "text-red-600";
    }
</script>

<div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
    <Card>
        <div class="p-6">
            <div class="text-sm font-medium text-gray-500 mb-1">Total Jobs</div>
            <div class="text-3xl font-bold text-gray-900">{totalJobs}</div>
        </div>
    </Card>

    <Card>
        <div class="p-6">
            <div class="text-sm font-medium text-gray-500 mb-1">
                Total Emissions
            </div>
            <div class="text-3xl font-bold {getEmissionsColor(totalEmissions)}">
                {formatNumber(totalEmissions, 3)} kg
            </div>
            <div class="text-xs text-gray-500 mt-1">CO₂ equivalent</div>
        </div>
    </Card>

    <Card>
        <div class="p-6">
            <div class="text-sm font-medium text-gray-500 mb-1">
                Total Energy
            </div>
            <div class="text-3xl font-bold text-gray-900">
                {formatNumber(totalEnergy, 2)} kWh
            </div>
        </div>
    </Card>

    <Card>
        <div class="p-6">
            <div class="text-sm font-medium text-gray-500 mb-1">
                Avg Carbon Intensity
            </div>
            <div class="text-3xl font-bold text-gray-900">
                {formatNumber(avgCarbonIntensity, 0)}
            </div>
            <div class="text-xs text-gray-500 mt-1">g CO₂/kWh</div>
        </div>
    </Card>
</div>
