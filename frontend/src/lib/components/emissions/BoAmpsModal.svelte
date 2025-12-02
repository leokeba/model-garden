<script lang="ts">
    import Button from "$lib/components/Button.svelte";
    import Card from "$lib/components/Card.svelte";

    interface Props {
        report: any;
        loading: boolean;
        onClose: () => void;
    }

    let { report, loading, onClose }: Props = $props();

    function downloadReport() {
        const blob = new Blob([JSON.stringify(report, null, 2)], {
            type: "application/json",
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `boamps-report-${report.header?.reportId || "report"}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }
</script>

{#if report}
    <div
        class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
    >
        <Card class="max-w-4xl w-full max-h-[80vh] flex flex-col">
            <div class="p-6 border-b border-gray-200">
                <div class="flex justify-between items-center">
                    <div>
                        <h2 class="text-2xl font-bold text-gray-900">
                            BoAmps Report
                        </h2>
                        <p class="text-sm text-gray-600 mt-1">
                            Standardized emissions report
                        </p>
                    </div>
                    <button
                        onclick={onClose}
                        class="text-gray-400 hover:text-gray-600"
                    >
                        ✕
                    </button>
                </div>
            </div>

            <div class="flex-1 overflow-y-auto p-6">
                {#if loading}
                    <div class="flex justify-center items-center h-32">
                        <div
                            class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"
                        ></div>
                    </div>
                {:else}
                    <pre
                        class="text-sm text-gray-900 whitespace-pre-wrap overflow-x-auto bg-gray-50 p-4 rounded-lg">{JSON.stringify(
                            report,
                            null,
                            2,
                        )}</pre>
                {/if}
            </div>

            <div class="p-6 border-t border-gray-200 flex gap-3">
                <Button onclick={downloadReport} variant="secondary">
                    📥 Download JSON
                </Button>
                <Button onclick={onClose} variant="secondary" fullWidth>
                    Close
                </Button>
            </div>
        </Card>
    </div>
{/if}
