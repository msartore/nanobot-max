package nanobot.java;

import com.gargoylesoftware.htmlunit.BrowserVersion;
import com.gargoylesoftware.htmlunit.WebClient;
import com.gargoylesoftware.htmlunit.html.HtmlPage;

import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Minimal HtmlUnit-based scraper.
 *
 * Usage: java -cp '/app/htmlunit/*' nanobot.java.Scraper <url> [jsWaitMs]
 *
 * Fetches the URL, waits for background JavaScript, then prints the page XML to stdout.
 * Exits non-zero on error; error messages go to stderr.
 */
public class Scraper {

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: Scraper <url> [jsWaitMs]");
            System.exit(1);
        }

        String url = args[0];
        int jsWaitMs = args.length > 1 ? Integer.parseInt(args[1]) : 3000;

        // Silence HtmlUnit's verbose logging so only real errors reach stderr
        silenceLogging();

        try (PrintStream out = new PrintStream(System.out, true, StandardCharsets.UTF_8);
             WebClient client = new WebClient(BrowserVersion.CHROME)) {

            configureClient(client, jsWaitMs);

            HtmlPage page = client.getPage(url);
            client.waitForBackgroundJavaScript(jsWaitMs);

            out.print(page.asXml());

        } catch (Exception e) {
            System.err.println("Error fetching " + url + ": " + e.getMessage());
            System.exit(2);
        }
    }

    private static void configureClient(WebClient client, int jsTimeoutMs) {
        client.getOptions().setJavaScriptEnabled(true);
        client.getOptions().setCssEnabled(false);
        client.getOptions().setThrowExceptionOnScriptError(false);
        client.getOptions().setThrowExceptionOnFailingStatusCode(false);
        client.getOptions().setPrintContentOnFailingStatusCode(false);
        client.getOptions().setRedirectEnabled(true);
        client.getOptions().setTimeout(30_000);
        client.setJavaScriptTimeout(jsTimeoutMs);
    }

    private static void silenceLogging() {
        String[] noisy = {
            "com.gargoylesoftware.htmlunit",
            "org.apache.http",
            "org.apache.commons",
        };
        for (String name : noisy) {
            Logger.getLogger(name).setLevel(Level.OFF);
        }
    }
}
