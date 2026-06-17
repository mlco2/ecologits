document.addEventListener('DOMContentLoaded', function() {
  const checkboxes = document.querySelectorAll('input.provider-option[type="checkbox"]');
  const commandOutput = document.getElementById('install-command');

  function updateCommand() {
    let selectedProviders = [];
    checkboxes.forEach(checkbox => {
      if (checkbox.checked) {
        selectedProviders.push(checkbox.value);
      }
    });

    let command = 'pip install ecologits';
    if (selectedProviders.length > 0) {
      command += `[${selectedProviders.join(',')}]`;
    }
    commandOutput.textContent = command;
  }

  checkboxes.forEach(checkbox => checkbox.addEventListener('change', updateCommand));

  // Initialize with default command
  updateCommand();
});