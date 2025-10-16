<script lang="ts">
  import { page } from '$app/stores';

  type NavItem = {
    name: string;
    href: string;
    icon: string;
  };

  const navigation: NavItem[] = [
    { name: 'Dashboard', href: '/', icon: '🏠' },
    { name: 'Models', href: '/models', icon: '📦' },
    { name: 'Training', href: '/training', icon: '🎓' },
    { name: 'Inference', href: '/inference', icon: '💬' },
    { name: 'Datasets', href: '/datasets', icon: '📊' },
    { name: 'Emissions', href: '/emissions', icon: '🌱' },
  ];

  const currentPath = $derived($page.url.pathname);

  function isActive(href: string): boolean {
    if (href === '/') {
      return currentPath === '/';
    }
    return currentPath.startsWith(href);
  }
</script>

<nav class="bg-white shadow-sm border-b border-gray-200">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
    <div class="flex justify-between h-16">
      <div class="flex">
        <!-- Logo -->
        <div class="flex-shrink-0 flex items-center">
          <a href="/" class="text-xl font-bold text-gray-900">
            🌱 Model Garden
          </a>
        </div>

        <!-- Navigation Links -->
        <div class="hidden sm:ml-8 sm:flex sm:space-x-8">
          {#each navigation as item}
            <a
              href={item.href}
              class="inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors {isActive(
                item.href
              )
                ? 'border-primary-500 text-gray-900'
                : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'}"
            >
              <span class="mr-2">{item.icon}</span>
              {item.name}
            </a>
          {/each}
        </div>
      </div>

      <!-- Right side buttons -->
      <div class="flex items-center space-x-4">
        <a
          href="/training/new"
          class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-colors"
        >
          + New Training
        </a>
      </div>
    </div>
  </div>

  <!-- Mobile menu -->
  <div class="sm:hidden">
    <div class="pt-2 pb-3 space-y-1">
      {#each navigation as item}
        <a
          href={item.href}
          class="block pl-3 pr-4 py-2 border-l-4 text-base font-medium {isActive(item.href)
            ? 'bg-primary-50 border-primary-500 text-primary-700'
            : 'border-transparent text-gray-600 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-800'}"
        >
          <span class="mr-2">{item.icon}</span>
          {item.name}
        </a>
      {/each}
    </div>
  </div>
</nav>
