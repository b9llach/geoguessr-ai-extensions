// ==UserScript==
// @name         AIMY Extension
// @namespace    http://tampermonkey.net/
// @version      1.0
// @description  Grab Pano ID and make a guess based off of image generated
// @author       billy
// @match        https://www.geoguessr.com/*
// @grant        none
// ==/UserScript==

(function() {
    'use strict';

    let globalPanoID = undefined;
    let roundNumber = 1;
    let isGuessing = false;  // Flag to prevent multiple guesses at the same time
    let previousGameID = null;

    function isNewGame() {
        const currentGameID = getGameID();
        if (currentGameID !== previousGameID) {
            previousGameID = currentGameID;
            return true;
        }
        return false;
    }

    function getGameID() {
        return window.location.pathname.split('/')[2];
    }

    async function wait(millis) {
        return new Promise((resolve) => setTimeout(resolve, millis));
    }

    async function getCoordinates(panoID) {
        const response = await fetch('http://127.0.0.1:6301/api/v1/predict', {
            method: 'POST',
            headers: {
                'accept': 'application/json',
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ panoID })
        });

        if (!response.ok) {
            const errorDetails = await response.text();
            throw new Error(`Failed to get coordinates: ${errorDetails}`);
        }

        const data = await response.json();
        return { lat: data.lat, lng: data.lng };
    }

    async function submitGuess(lat, lng, roundNumber) {
        const gameID = getGameID();
        let apiURL = `https://game-server.geoguessr.com/api/duels/${gameID}/guess`;
        if (window.location.href.includes('battle-royale')) {
            apiURL = `https://game-server.geoguessr.com/api/battle-royale/${gameID}/guess`;
        }

        const payload = { lat, lng, roundNumber };
        const headers = {
            origin: "https://www.geoguessr.com",
            referer: apiURL,
            "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
            "sec-ch-ua": '"Chromium";v="106", "Google Chrome";v="106", "Not;A=Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "macOS",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36",
            "x-client": "web",
            "Content-Type": "application/json",
        };
        await wait(6000); // Beginning of round
        for (let i=0; i<5; i++) {
            try {
                const response = await fetch(apiURL, {
                    method: "POST",
                    credentials: "include",
                    headers: headers,
                    body: JSON.stringify(payload),
                });

                if (!response.ok) {
                    const errorDetails = await response.text();
                    throw new Error(`Guess submission failed: ${errorDetails}`);
                }

                const body = await response.json();
                return { resp: response, body: body };
            } catch (error) {
                console.error("Error submitting guess:", error);
                await wait(3000);
            }
        }
        throw new Error("Failed to submit guess after 5 attempts");
    }

    var originalOpen = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function(method, url) {

        if (method.toUpperCase() === 'POST' &&
            (url.startsWith('https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/GetMetadata') ||
             url.startsWith('https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/SingleImageSearch'))) {

            this.addEventListener('load', async function () {
                if (isGuessing) return;  // Skip if a guess is already in progress
                isGuessing = true;
                try {
                    let interceptedResult = this.responseText;
                    let jsonResponse = JSON.parse(interceptedResult);

                    globalPanoID = jsonResponse[1][0][1][1];

                    if (globalPanoID) {
                        console.log(`Round ${roundNumber}: ${globalPanoID}`);
                    }

                    try {
                        const { lat, lng } = await getCoordinates(globalPanoID);
                        console.log(`Guessing (${lat}, ${lng})`);

                        try {
                            const result = await submitGuess(lat, lng, roundNumber);
                            console.log(`Guess submitted for round ${roundNumber}:`, result);
                        } catch (error) {
                            console.error(`Failed to submit guess for round ${roundNumber}:`, error);
                        }

                    } catch (error) {
                        console.error(`Failed to get coordinates for panoID ${globalPanoID}:`, error);
                    }
                } catch (error) {
                    console.error('Error parsing JSON response:', error);
                }

                roundNumber++;
                isGuessing = false;
            });
        }
        return originalOpen.apply(this, arguments);
    };

    var originalFetch = window.fetch;
    window.fetch = function() {
        return originalFetch.apply(this, arguments).then(function(response) {
            if (response.url.includes('/duels/')) {
                if (isNewGame()) {
                    roundNumber = 1;
                    console.log('New game started. Round number reset to 1.');
                }
            }
            return response;
        });
    };
})();
