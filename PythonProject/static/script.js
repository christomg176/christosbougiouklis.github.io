function askAI() {
    const prompt = document.getElementById('prompt').value;
    if (!prompt) return;

    fetch('/api/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        const messages = document.getElementById('messages');
        messages.innerHTML += `<div class="message">${data.response}</div>`;
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Failed to get AI response. See console for details.');
    });
}