<script lang="ts">
    import {
        api,
        type SystemSettings,
        type TrainingBackend,
    } from "$lib/api/client";
    import Badge from "$lib/components/Badge.svelte";
    import Button from "$lib/components/Button.svelte";
    import Card from "$lib/components/Card.svelte";
    import { onDestroy, onMount } from "svelte";

    let settings: SystemSettings | null = $state(null);
    let backends: TrainingBackend[] = $state([]);
    let loading = $state(true);
    let error = $state("");

    let carbonPower = $state(0);
    let carbonDuration = $state(0);
    let carbonSaving = $state(false);
    let carbonSaveMessage = $state("");

    let publisherName = $state("");
    let division = $state("");
    let defaultProject = $state("");
    let infraType = $state("");
    let locationCountry = $state("");
    let locationRegion = $state("");
    let reportSaving = $state(false);
    let reportSaveMessage = $state("");

    // Operation states
    let operationInProgress = $state(false);
    let operationMessage = $state("");
    let showRestartConfirm = $state(false);
    let showUninstallConfirm = $state(false);
    let restartRequired = $state(false);
    let pollingInterval: number | null = null;

    async function loadSettings() {
        try {
            const [settingsResponse, backendsResponse] = await Promise.all([
                api.getSettings(),
                api.getBackends(),
            ]);
            settings = settingsResponse;
            backends = backendsResponse.backends;

            carbonPower = settingsResponse.carbon?.power_calibration_kwh ?? 0;
            carbonDuration =
                settingsResponse.carbon?.duration_calibration_seconds ?? 0;

            publisherName = settingsResponse.report?.publisher_name ?? "";
            division = settingsResponse.report?.division ?? "";
            defaultProject =
                settingsResponse.report?.default_project_name ?? "";
            infraType = settingsResponse.report?.infra_type ?? "";
            locationCountry = settingsResponse.report?.location_country ?? "";
            locationRegion = settingsResponse.report?.location_region ?? "";

            // Check if there's an ongoing operation
            if (settings.package_operation.in_progress) {
                startPolling();
            }
        } catch (err) {
            error =
                err instanceof Error ? err.message : "Failed to load settings";
        } finally {
            loading = false;
        }
    }

    async function handleSaveCarbonSettings() {
        if (!settings) return;
        carbonSaving = true;
        carbonSaveMessage = "";
        try {
            const updated = await api.updateSettings({
                carbon: {
                    power_calibration_kwh: Number(carbonPower),
                    duration_calibration_seconds: Number(carbonDuration),
                },
            });
            settings = updated;
            carbonSaveMessage = "Saved";
        } catch (err) {
            carbonSaveMessage =
                err instanceof Error ? err.message : "Failed to save";
        } finally {
            carbonSaving = false;
        }
    }

    async function handleSaveReportSettings() {
        if (!settings) return;
        reportSaving = true;
        reportSaveMessage = "";
        try {
            const updated = await api.updateSettings({
                report: {
                    publisher_name: publisherName,
                    division,
                    default_project_name: defaultProject,
                    infra_type: infraType,
                    location_country: locationCountry,
                    location_region: locationRegion,
                },
            });
            settings = updated;
            reportSaveMessage = "Saved";
        } catch (err) {
            reportSaveMessage =
                err instanceof Error ? err.message : "Failed to save";
        } finally {
            reportSaving = false;
        }
    }

    function startPolling() {
        if (pollingInterval) return;
        pollingInterval = setInterval(async () => {
            try {
                const response = await api.getUnslothOperationStatus();
                if (settings) {
                    settings.package_operation = response.data;
                }

                // Stop polling if operation completed
                if (!response.data.in_progress) {
                    stopPolling();
                    // Mark that restart is required for changes to take effect
                    if (response.data.success) {
                        restartRequired = true;
                    }
                    // Refresh settings
                    await loadSettings();
                }
            } catch (err) {
                console.error("Polling error:", err);
            }
        }, 2000);
    }

    function stopPolling() {
        if (pollingInterval) {
            clearInterval(pollingInterval);
            pollingInterval = null;
        }
    }

    async function handleInstallUnsloth() {
        operationInProgress = true;
        operationMessage = "";
        try {
            const response = await api.installUnsloth();
            if (response.success) {
                operationMessage = response.message;
                startPolling();
            } else {
                operationMessage = response.message;
            }
        } catch (err) {
            operationMessage =
                err instanceof Error
                    ? err.message
                    : "Failed to install Unsloth";
        } finally {
            operationInProgress = false;
        }
    }

    async function handleUninstallUnsloth() {
        showUninstallConfirm = false;
        operationInProgress = true;
        operationMessage = "";
        try {
            const response = await api.uninstallUnsloth();
            if (response.success) {
                operationMessage = response.message;
                startPolling();
            } else {
                operationMessage = response.message;
            }
        } catch (err) {
            operationMessage =
                err instanceof Error
                    ? err.message
                    : "Failed to uninstall Unsloth";
        } finally {
            operationInProgress = false;
        }
    }

    async function handleRestartService() {
        showRestartConfirm = false;
        operationInProgress = true;
        operationMessage = "Restarting service...";

        try {
            await api.restartService();
        } catch (err) {
            // Connection errors are expected - the service restarts before responding
            // Only show error if it's a real API error (like permission denied)
            const errorMsg = err instanceof Error ? err.message : String(err);
            if (
                errorMsg.includes("403") ||
                errorMsg.includes("sudo") ||
                errorMsg.includes("permission")
            ) {
                operationMessage = errorMsg;
                operationInProgress = false;
                return;
            }
            // Otherwise, assume the restart was initiated and connection was lost
        }

        operationMessage =
            "Service restart initiated. Waiting for service to come back up...";

        // Wait a bit then try to reconnect
        setTimeout(() => {
            checkServiceAndReload();
        }, 3000);
    }

    async function checkServiceAndReload() {
        let attempts = 0;
        const maxAttempts = 30;

        const checkHealth = async () => {
            try {
                await api.getHealth();
                // Service is back up, reload the page
                window.location.reload();
            } catch (err) {
                attempts++;
                if (attempts < maxAttempts) {
                    operationMessage = `Waiting for service to restart... (${attempts}/${maxAttempts})`;
                    setTimeout(checkHealth, 2000);
                } else {
                    operationMessage =
                        "Service may still be restarting. Please refresh the page manually.";
                    operationInProgress = false;
                }
            }
        };

        checkHealth();
    }

    onMount(() => {
        loadSettings();
    });

    onDestroy(() => {
        stopPolling();
    });
</script>

<svelte:head>
    <title>Settings - Model Garden</title>
</svelte:head>

<div class="min-h-screen bg-gray-50 pt-6">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <!-- Header -->
        <div class="mb-8">
            <h1 class="text-3xl font-bold text-gray-900">Settings</h1>
            <p class="mt-2 text-sm text-gray-600">
                Manage optional dependencies and system configuration
            </p>
        </div>

        {#if loading}
            <div class="text-center py-12">
                <div
                    class="inline-block w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full animate-spin"
                ></div>
                <p class="mt-2 text-gray-600">Loading settings...</p>
            </div>
        {:else if error}
            <div class="text-center py-12">
                <div class="text-red-600 text-lg">{error}</div>
                <Button
                    onclick={() => window.location.reload()}
                    variant="primary"
                    class="mt-4"
                >
                    Retry
                </Button>
            </div>
        {:else if settings}
            <!-- Operation Status Banner -->
            {#if settings.package_operation.in_progress || operationMessage}
                <Card class="mb-6 border-l-4 border-l-blue-500">
                    <div class="flex items-start gap-4">
                        {#if settings.package_operation.in_progress}
                            <div
                                class="w-6 h-6 border-3 border-blue-600 border-t-transparent rounded-full animate-spin flex-shrink-0 mt-0.5"
                            ></div>
                        {:else if settings.package_operation.success === true}
                            <span class="text-2xl flex-shrink-0">✅</span>
                        {:else if settings.package_operation.success === false}
                            <span class="text-2xl flex-shrink-0">❌</span>
                        {:else}
                            <span class="text-2xl flex-shrink-0">ℹ️</span>
                        {/if}
                        <div class="flex-1 min-w-0">
                            <p class="font-medium text-gray-900">
                                {#if settings.package_operation.in_progress}
                                    {settings.package_operation.operation ===
                                    "install_unsloth"
                                        ? "Installing Unsloth..."
                                        : "Uninstalling Unsloth..."}
                                {:else if operationMessage}
                                    {operationMessage}
                                {:else if settings.package_operation.success === true}
                                    Operation completed successfully
                                {:else if settings.package_operation.success === false}
                                    Operation failed: {settings
                                        .package_operation.error}
                                {/if}
                            </p>
                            {#if settings.package_operation.output.length > 0}
                                <details class="mt-2">
                                    <summary
                                        class="cursor-pointer text-sm text-gray-500 hover:text-gray-700"
                                        >Show output</summary
                                    >
                                    <pre
                                        class="mt-2 p-3 bg-gray-900 text-gray-100 text-xs rounded-lg overflow-x-auto max-h-48 overflow-y-auto">{settings.package_operation.output.join(
                                            "\n",
                                        )}</pre>
                                </details>
                            {/if}
                            {#if restartRequired && !settings.package_operation.in_progress}
                                <div
                                    class="mt-3 p-3 bg-amber-50 border border-amber-200 rounded-lg"
                                >
                                    <p class="text-sm text-amber-800 mb-2">
                                        <strong>Restart required:</strong> The service
                                        needs to be restarted for changes to take
                                        effect.
                                    </p>
                                    <Button
                                        variant="primary"
                                        size="sm"
                                        onclick={() => {
                                            restartRequired = false;
                                            showRestartConfirm = true;
                                        }}
                                    >
                                        Restart Now
                                    </Button>
                                </div>
                            {/if}
                        </div>
                    </div>
                </Card>
            {/if}

            <!-- Optional Dependencies Section -->
            <div class="mb-8">
                <h2 class="text-xl font-semibold text-gray-900 mb-4">
                    Optional Dependencies
                </h2>
                <Card>
                    <!-- Unsloth -->
                    <div class="flex items-start justify-between">
                        <div class="flex-1">
                            <div class="flex items-center gap-3 mb-2">
                                <h3 class="text-lg font-medium text-gray-900">
                                    Unsloth
                                </h3>
                                {#if settings.optional_dependencies.unsloth.installed}
                                    <Badge variant="success" size="sm"
                                        >Installed</Badge
                                    >
                                    {#if settings.optional_dependencies.unsloth.version}
                                        <span class="text-xs text-gray-500"
                                            >v{settings.optional_dependencies
                                                .unsloth.version}</span
                                        >
                                    {/if}
                                {:else}
                                    <Badge variant="warning" size="sm"
                                        >Not Installed</Badge
                                    >
                                {/if}
                            </div>
                            <p class="text-sm text-gray-600">
                                {settings.optional_dependencies.unsloth
                                    .description}
                            </p>
                            {#if !settings.optional_dependencies.unsloth.installed}
                                <p class="text-xs text-gray-500 mt-2">
                                    Installing Unsloth enables the optimized
                                    training backend for faster fine-tuning with
                                    lower memory usage.
                                </p>
                            {/if}
                        </div>
                        <div class="ml-4 flex-shrink-0">
                            {#if settings.optional_dependencies.unsloth.installed}
                                <Button
                                    variant="danger"
                                    size="sm"
                                    onclick={() =>
                                        (showUninstallConfirm = true)}
                                    disabled={settings.package_operation
                                        .in_progress || operationInProgress}
                                >
                                    Uninstall
                                </Button>
                            {:else}
                                <Button
                                    variant="primary"
                                    size="sm"
                                    onclick={handleInstallUnsloth}
                                    disabled={settings.package_operation
                                        .in_progress || operationInProgress}
                                    loading={settings.package_operation
                                        .in_progress &&
                                        settings.package_operation.operation ===
                                            "install_unsloth"}
                                >
                                    Install
                                </Button>
                            {/if}
                        </div>
                    </div>
                </Card>
            </div>

            <!-- Training Backends Section -->
            <div class="mb-8">
                <h2 class="text-xl font-semibold text-gray-900 mb-4">
                    Training Backends
                </h2>
                <Card>
                    <div class="space-y-4">
                        {#each backends as backend}
                            <div
                                class="flex items-start justify-between py-3 border-b border-gray-100 last:border-0"
                            >
                                <div>
                                    <div class="flex items-center gap-2 mb-1">
                                        <h4
                                            class="font-medium text-gray-900 capitalize"
                                        >
                                            {backend.name}
                                        </h4>
                                        {#if backend.name === "unsloth" && settings.optional_dependencies.unsloth.installed}
                                            <Badge variant="success" size="sm"
                                                >Active</Badge
                                            >
                                        {:else if backend.name === "transformers"}
                                            <Badge variant="info" size="sm"
                                                >Default</Badge
                                            >
                                        {/if}
                                    </div>
                                    <p class="text-sm text-gray-600">
                                        {backend.description}
                                    </p>
                                </div>
                                <div class="flex gap-2 text-xs">
                                    {#if backend.supports_text}
                                        <span
                                            class="px-2 py-1 bg-blue-100 text-blue-700 rounded-full"
                                            >Text</span
                                        >
                                    {/if}
                                    {#if backend.supports_vision}
                                        <span
                                            class="px-2 py-1 bg-purple-100 text-purple-700 rounded-full"
                                            >Vision</span
                                        >
                                    {/if}
                                </div>
                            </div>
                        {/each}
                    </div>
                </Card>
            </div>

            <!-- Carbon Calibration Section -->
            <div class="mb-8">
                <h2 class="text-xl font-semibold text-gray-900 mb-4">
                    Carbon Calibration
                </h2>
                <Card>
                    <div class="space-y-4">
                        <p class="text-sm text-gray-600">
                            Provide a baseline measurement to subtract idle
                            power from task emissions. Use a short calibration
                            run before jobs and record its energy (kWh) and
                            duration (seconds).
                        </p>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label
                                    class="block text-sm font-medium text-gray-700 mb-1"
                                    for="power-calibration"
                                    >Power calibration (kWh)</label
                                >
                                <input
                                    id="power-calibration"
                                    type="number"
                                    step="0.0001"
                                    min="0"
                                    class="w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
                                    bind:value={carbonPower}
                                />
                            </div>
                            <div>
                                <label
                                    class="block text-sm font-medium text-gray-700 mb-1"
                                    for="duration-calibration"
                                    >Calibration duration (seconds)</label
                                >
                                <input
                                    id="duration-calibration"
                                    type="number"
                                    step="1"
                                    min="0"
                                    class="w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
                                    bind:value={carbonDuration}
                                />
                            </div>
                        </div>
                        <div class="flex items-center gap-3">
                            <Button
                                variant="primary"
                                size="sm"
                                onclick={handleSaveCarbonSettings}
                                disabled={carbonSaving}
                                loading={carbonSaving}
                            >
                                Save Calibration
                            </Button>
                            {#if carbonSaveMessage}
                                <span class="text-sm text-gray-600">
                                    {carbonSaveMessage}
                                </span>
                            {/if}
                        </div>
                    </div>
                </Card>
            </div>

            <!-- Report Defaults Section -->
            <div class="mb-8">
                <h2 class="text-xl font-semibold text-gray-900 mb-4">
                    Report Defaults
                </h2>
                <Card>
                    <div class="space-y-4">
                        <p class="text-sm text-gray-600">
                            Defaults applied to generated reports. These values
                            can be overridden per job, but setting them here
                            keeps metadata consistent.
                        </p>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label
                                    class="block text-sm font-medium text-gray-700 mb-1"
                                    for="publisher-name">Publisher name</label
                                >
                                <input
                                    id="publisher-name"
                                    type="text"
                                    class="w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
                                    bind:value={publisherName}
                                />
                            </div>
                            <div>
                                <label
                                    class="block text-sm font-medium text-gray-700 mb-1"
                                    for="division">Division / team</label
                                >
                                <input
                                    id="division"
                                    type="text"
                                    class="w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
                                    bind:value={division}
                                />
                            </div>
                            <div>
                                <label
                                    class="block text-sm font-medium text-gray-700 mb-1"
                                    for="project-name"
                                    >Default project name</label
                                >
                                <input
                                    id="project-name"
                                    type="text"
                                    class="w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
                                    bind:value={defaultProject}
                                />
                            </div>
                            <div>
                                <label
                                    class="block text-sm font-medium text-gray-700 mb-1"
                                    for="infra-type"
                                    >Infra type (onPremise / publicCloud /
                                    privateCloud)</label
                                >
                                <input
                                    id="infra-type"
                                    type="text"
                                    class="w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
                                    bind:value={infraType}
                                />
                            </div>
                            <div>
                                <label
                                    class="block text-sm font-medium text-gray-700 mb-1"
                                    for="location-country"
                                    >Location country</label
                                >
                                <input
                                    id="location-country"
                                    type="text"
                                    class="w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
                                    bind:value={locationCountry}
                                />
                            </div>
                            <div>
                                <label
                                    class="block text-sm font-medium text-gray-700 mb-1"
                                    for="location-region"
                                    >Location region / city</label
                                >
                                <input
                                    id="location-region"
                                    type="text"
                                    class="w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
                                    bind:value={locationRegion}
                                />
                            </div>
                        </div>
                        <div class="flex items-center gap-3">
                            <Button
                                variant="primary"
                                size="sm"
                                onclick={handleSaveReportSettings}
                                disabled={reportSaving}
                                loading={reportSaving}
                            >
                                Save Report Defaults
                            </Button>
                            {#if reportSaveMessage}
                                <span class="text-sm text-gray-600">
                                    {reportSaveMessage}
                                </span>
                            {/if}
                        </div>
                    </div>
                </Card>
            </div>

            <!-- Service Management Section -->
            <div class="mb-8">
                <h2 class="text-xl font-semibold text-gray-900 mb-4">
                    Service Management
                </h2>
                <Card>
                    <div class="flex items-start justify-between">
                        <div class="flex-1">
                            <div class="flex items-center gap-3 mb-2">
                                <h3 class="text-lg font-medium text-gray-900">
                                    Model Garden Service
                                </h3>
                                {#if settings.service.is_systemd_service}
                                    <Badge variant="success" size="sm"
                                        >Running</Badge
                                    >
                                {:else}
                                    <Badge variant="warning" size="sm"
                                        >Manual Mode</Badge
                                    >
                                {/if}
                            </div>
                            {#if settings.service.is_systemd_service}
                                <p class="text-sm text-gray-600">
                                    Running as a systemd service. Restart the
                                    service to apply changes to installed
                                    packages.
                                </p>
                                {#if !settings.service.can_restart_service}
                                    <p class="text-xs text-amber-600 mt-2">
                                        ⚠️ Passwordless sudo not configured. To
                                        enable restart from the UI, add to <code
                                            class="bg-gray-100 px-1 rounded"
                                            >/etc/sudoers.d/model-garden</code
                                        >:
                                    </p>
                                    <pre
                                        class="text-xs bg-gray-100 p-2 rounded mt-1 overflow-x-auto">&lt;username&gt; ALL=(root) NOPASSWD: /usr/bin/systemctl restart model-garden.service</pre>
                                {/if}
                            {:else}
                                <p class="text-sm text-gray-600">
                                    Running in manual mode. Restart the server
                                    manually (Ctrl+C and re-run) to apply
                                    changes.
                                </p>
                            {/if}
                        </div>
                        <div class="ml-4 flex-shrink-0">
                            <Button
                                variant="warning"
                                size="sm"
                                onclick={() => (showRestartConfirm = true)}
                                disabled={!settings.service
                                    .is_systemd_service ||
                                    !settings.service.can_restart_service ||
                                    operationInProgress}
                                title={!settings.service.is_systemd_service
                                    ? "Not running as a systemd service"
                                    : !settings.service.can_restart_service
                                      ? "Passwordless sudo not configured"
                                      : "Restart the service"}
                            >
                                Restart Service
                            </Button>
                        </div>
                    </div>
                </Card>
            </div>

            <!-- Environment Info Section -->
            <div class="mb-8">
                <h2 class="text-xl font-semibold text-gray-900 mb-4">
                    Environment
                </h2>
                <Card>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                            <span class="text-sm text-gray-500"
                                >Python Version</span
                            >
                            <p class="font-medium text-gray-900">
                                {settings.environment.python_version}
                            </p>
                        </div>
                        <div>
                            <span class="text-sm text-gray-500"
                                >Transformers Version</span
                            >
                            <p class="font-medium text-gray-900">
                                {settings.environment.transformers_version}
                            </p>
                        </div>
                        <div>
                            <span class="text-sm text-gray-500"
                                >Project Root</span
                            >
                            <p
                                class="font-medium text-gray-900 text-sm truncate"
                                title={settings.environment.project_root}
                            >
                                {settings.environment.project_root}
                            </p>
                        </div>
                    </div>
                </Card>
            </div>
        {/if}
    </div>
</div>

<!-- Uninstall Confirmation Modal -->
{#if showUninstallConfirm}
    <div
        class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
    >
        <Card class="max-w-md mx-4">
            <div class="text-center">
                <div class="text-4xl mb-4">⚠️</div>
                <h3 class="text-lg font-semibold text-gray-900 mb-2">
                    Uninstall Unsloth?
                </h3>
                <p class="text-gray-600 text-sm mb-6">
                    This will remove Unsloth and reinstall the latest version of
                    Transformers (which Unsloth constrains to older versions).
                    You'll need to use the Transformers backend for training
                    until Unsloth is reinstalled.
                </p>
                <div class="flex gap-3 justify-center">
                    <Button
                        variant="secondary"
                        onclick={() => (showUninstallConfirm = false)}
                    >
                        Cancel
                    </Button>
                    <Button variant="danger" onclick={handleUninstallUnsloth}>
                        Uninstall
                    </Button>
                </div>
            </div>
        </Card>
    </div>
{/if}

<!-- Restart Confirmation Modal -->
{#if showRestartConfirm}
    <div
        class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
    >
        <Card class="max-w-md mx-4">
            <div class="text-center">
                <div class="text-4xl mb-4">🔄</div>
                <h3 class="text-lg font-semibold text-gray-900 mb-2">
                    Restart Service?
                </h3>
                <p class="text-gray-600 text-sm mb-6">
                    The service will be restarted. Any running training jobs
                    will be interrupted. The page will automatically reconnect
                    when the service is back up.
                </p>
                <div class="flex gap-3 justify-center">
                    <Button
                        variant="secondary"
                        onclick={() => (showRestartConfirm = false)}
                    >
                        Cancel
                    </Button>
                    <Button variant="warning" onclick={handleRestartService}>
                        Restart
                    </Button>
                </div>
            </div>
        </Card>
    </div>
{/if}
